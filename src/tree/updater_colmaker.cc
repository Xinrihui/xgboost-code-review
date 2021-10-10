/*!
 * Copyright 2014-2019 by Contributors
 * \file updater_.cc
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <rabit/rabit.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

#include "xgboost/parameter.h"
#include "xgboost/tree_updater.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "param.h"
#include "constraints.h"
#include "../common/random.h"
#include "split_evaluator.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_);

struct TrainParam : XGBoostParameter<TrainParam> {
  // speed optimization for dense column
  float opt_dense_col;
  DMLC_DECLARE_PARAMETER(TrainParam) {
    DMLC_DECLARE_FIELD(opt_dense_col)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("EXP Param: speed optimization for dense column.");
  }

  /*! \brief whether need forward small to big search: default right */
  inline bool NeedForwardSearch(int default_direction, float col_density,
                                bool indicator) const {
    return default_direction == 2 ||
           (default_direction == 0 && (col_density < opt_dense_col) &&
            !indicator);
  }
  /*! \brief whether need backward big to small search: default left */
  inline bool NeedBackwardSearch(int default_direction) const {
    return default_direction != 2;
  }
};

DMLC_REGISTER_PARAMETER(TrainParam);

/*! \brief column-wise update to construct a tree */
class : public TreeUpdater {
 public:
  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    _param_.UpdateAllowUnknown(args);
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    FromJson(config.at("_train_param"), &this->_param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["_train_param"] = ToJson(_param_);
  }

  char const* Name() const override {
    return "grow_";
  }

  void LazyGetColumnDensity(DMatrix *dmat) {
    // Finds densities if we don't already have them
    if (column_densities_.empty()) {
      std::vector<size_t> column_size(dmat->Info().num_col_);
      for (const auto &batch : dmat->GetBatches<SortedCSCPage>()) {
        auto page = batch.GetView();
        for (auto i = 0u; i < batch.Size(); i++) {
          column_size[i] += page[i].size();
        }
      }
      column_densities_.resize(column_size.size());
      for (auto i = 0u; i < column_densities_.size(); i++) {
        size_t nmiss = dmat->Info().num_row_ - column_size[i];
        column_densities_[i] =
            1.0f - (static_cast<float>(nmiss)) / dmat->Info().num_row_;
      }
    }
  }

  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    if (rabit::IsDistributed()) {
      LOG(FATAL) << "Updater `grow_` or `exact` tree method doesn't "
                    "support distributed training.";
    }
    this->LazyGetColumnDensity(dmat);
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    interaction_constraints_.Configure(param_, dmat->Info().num_row_);
    // build tree
    for (auto tree : trees) {
      Builder builder(
        param_,
        _param_,
        interaction_constraints_, column_densities_);
      builder.Update(gpair->ConstHostVector(), dmat, tree);
    }
    param_.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param_;
  TrainParam _param_;
  // SplitEvaluator that will be cloned for each Builder
  std::vector<float> column_densities_;

  FeatureInteractionConstraintHost interaction_constraints_;
 
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    GradStats stats;

    /*! \brief last feature value scanned */
    //最后扫描到的特征值
    bst_float last_fvalue { 0 };

    /*! \brief current best solution */
    //最优的分裂方案(分裂特征index，分裂值，增益loss变化)
    SplitEntry best;

    // constructor
    ThreadEntry() = default;
  };

  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;

    /*! \brief loss of this node, without split */
    //节点没有分裂时的增益
    bst_float root_gain { 0.0f };

    /*! \brief weight calculated related to current data */
    // 当前计算的最优weight
    bst_float weight { 0.0f };
    
    /*! \brief current best solution */
    //最优的分裂方案(分裂特征index，分裂值，增益loss变化)
    SplitEntry best;
    
    // constructor
    NodeEntry() = default;
  };

  // actual builder that runs the algorithm
  class Builder {

   public:

    // constructor
    explicit Builder(const TrainParam& param,
                     const TrainParam& _train_param,
                     FeatureInteractionConstraintHost _interaction_constraints,
                     const std::vector<float> &column_densities)
        : param_(param), _train_param_{_train_param},
          nthread_(omp_get_max_threads()), 
          tree_evaluator_(param_, column_densities.size(), GenericParameter::kCpuId),
          interaction_constraints_{std::move(_interaction_constraints)},
          column_densities_(column_densities) {}

    // update one tree, growing
    virtual void Update(const std::vector<GradientPair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      std::vector<int> newnodes;

      //核心代码：
      
      //初始化数据和节点的映射关系，根据配置进行样本的降采样(伯努利采样)
      //qexpand_用于存储每次探索出候选树节点，初始化为root节点。
      this->InitData(gpair, *p_fmat);
      
      //计算qexpand_队列中所有候选节点的损失函数和权重
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
      
      //根据树的最大深度进行生长
      //根据参数param.max_depth逐层分裂生成节点和查找分裂最优方案
      for (int depth = 0; depth < param_.max_depth; ++depth) {
        
        //查找最佳分裂点
        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);

        //根据分裂结果，将数据重新映射到子节点 , 即更新数据样本到树节点的映射关系, 处理缺失值样本;
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        
        //将待扩展分割的叶子结点用于替换 qexpand_，作为下一轮split的候选节点
        this->UpdateQueueExpand(*p_tree, qexpand_, &newnodes);
        
        //重新初始化，计算 newnodes队列中所有候选节点的损失函数和权重.主要是为了下层训练的开始。
        this->InitNewNode(newnodes, gpair, *p_fmat, *p_tree);

        for (auto nid : qexpand_) {
          if ((*p_tree)[nid].IsLeaf()) {
            continue;
          }
          int cleft = (*p_tree)[nid].LeftChild();
          int cright = (*p_tree)[nid].RightChild();

          tree_evaluator_.AddSplit(nid, cleft, cright, snode_[nid].best.SplitIndex(),
                                   snode_[cleft].weight, snode_[cright].weight);
          interaction_constraints_.Split(nid, snode_[nid].best.SplitIndex(), cleft, cright);
        }
        qexpand_ = newnodes;
        
        // if nothing left to be expand, break
        //若无需继续分裂，则停止
        if (qexpand_.size() == 0) break;

      }

      // set all the rest expanding nodes to leaf
      // 对于达到训练深度但是还没结束时，将多余的expand节点都设置为不更新叶子节点状态，并设置最优weight值。并将所有节点在builder对象中上统计数据同步到RegTree对象上。
      for (const int nid : qexpand_) {
        (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
        p_tree->Stat(nid).base_weight = snode_[nid].weight;
        p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
      }
    }

   protected:

    // initialize temp data structure
    inline void InitData(const std::vector<GradientPair>& gpair,
                         const DMatrix& fmat) {
      {
        // setup position
        // 1）初始化每个样本实例开始所在root节点位置 position，如果是单任务，则root节点为0 position_[i]=0  ；如果是多任务，对应root信息为taskid。
        position_.resize(gpair.size()); // position_ 数组的长度为 总数据集的样本个数
                                        //  eg. position_[1]=0 : 1号 样本位于 0 号节点
        CHECK_EQ(fmat.Info().num_row_, position_.size());
        std::fill(position_.begin(), position_.end(), 0); // position_[i]=0
        
        // mark delete for the deleted datas
        //  2）删除二阶梯度小于0的样本实例，直接 position取反，最高位为1( position[i]的值为负数 ) ，则实例在将来分裂统计会被跳过。
        for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
          if (gpair[ridx].GetHess() < 0.0f) position_[ridx] = ~position_[ridx];
        }
        
        // mark subsample
        // 3）基于随机森林的行采样特性，利用伯努利来采样比例 param.subsample作为训练初始数据，删除实例也是 position取反。
        if (param_.subsample < 1.0f) {
          CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
            << "Only uniform sampling is supported, "
            << "gradient-based sampling is only support by GPU Hist.";
          std::bernoulli_distribution coin_flip(param_.subsample);
          auto& rnd = common::GlobalRandom();
          for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
            if (gpair[ridx].GetHess() < 0.0f) continue;
            if (!coin_flip(rnd)) position_[ridx] = ~position_[ridx];
          }
        }
      }

      // 4）利用 colsample_bytree参数，不同于随机森林列采样，后面会有真正的随机列采样过程，
      //   利用伯努利采样生成回归树训练初始特征候选集，构建生成 feat_index，index为序号，value为特征id。
      {
        column_sampler_.Init(fmat.Info().num_col_,
                             fmat.Info().feature_weigths.ConstHostVector(),
                             param_.colsample_bynode, param_.colsample_bylevel,
                             param_.colsample_bytree);
      }

      {
        // setup temp space for each thread
        // reserve a small space
        // 5）为线程计算初始化临时空间 stemp，每个空间预设256个 ThreadEntry，预设256个统计节点空间 snode，预设 256个 qexpand_元素，同时设置第一层分裂节点为 0到 param.num_roots-1。 
        // stemp、snode都是基于生成所有节点， qexpand_为当前待分裂节点，树深在7层以内不需要额外 vector内存分配，超过7层会先引起 stemp、snode重新分配。
        stemp_.clear();
        stemp_.resize(this->nthread_, std::vector<ThreadEntry>());
        for (auto& i : stemp_) {
          i.clear(); i.reserve(256);
        }
        snode_.reserve(256);
      }

      {
        // expand query
        qexpand_.reserve(256); qexpand_.clear();
        qexpand_.push_back(0);
      }
    }


    /*!
     * \brief initialize the base_weight, root_gain,
     *  and NodeEntry for all the new nodes in qexpand
     */
    // 初始化 qexpand 待分裂节点的统计信息
    inline void InitNewNode(const std::vector<int>& qexpand,
                            const std::vector<GradientPair>& gpair,
                            const DMatrix& fmat,
                            const RegTree& tree) {
      {
        // setup statistics space for each tree node
        //1）初始化qexpand_待分裂节点对应index在 stemp、snode的梯度统计信息与constraints_的预设信息。
        for (auto& i : stemp_) {
          i.resize(tree.param.num_nodes, ThreadEntry());
        }
        snode_.resize(tree.param.num_nodes, NodeEntry());
      }

      const MetaInfo& info = fmat.Info();

      // setup position
      // 2）利用OMP多线程 按行分片，并行统计训练数据所在节点, position[ridx]对应的统计信息，
      //    position<0 表示删除节点或者不进行继续分裂的节点，存在后者是因为xgboost是按照level逐层进行分裂查找，每层的数据是全量数据，
      //    按照position来分配到expand_分裂节点id上，对于早期某层节点无法继续分裂情况，会对该节点的所有的实例设置 position<0，因此需要对这部分实例进行过滤处理。
      const auto ndata = static_cast<bst_omp_uint>(info.num_row_);
      dmlc::OMPException exc;
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint ridx = 0; ridx < ndata; ++ridx) { // 开多线程遍历 所有的 训练数据 gpair , ridx - 训练数据的行索引

        exc.Run([&]() {

          const int tid = omp_get_thread_num();//omp_get_thread_num()用于获取当前线程在当前线程组中的序号；
          if (position_[ridx] < 0) return;
          stemp_[tid][position_[ridx]].stats.Add(gpair[ridx]); // position_[ridx] 为树节点的 标号; 初始状况时只有根节点, position_[ridx]=0

        });

      }
      exc.Rethrow();

      // sum the per thread statistics together
      // 3）合并多线程梯度统计信息到snode；设置约束条件，Colmaker基于NoConstraint，
      //    实际上不会执行任何约束设置操作；
      for (int nid : qexpand) {
        GradStats stats;
        for (auto& s : stemp_) {
          stats.Add(s[nid].stats);
        }

        // update node statistics
        snode_[nid].stats = stats;
      }

      auto evaluator = tree_evaluator_.GetEvaluator();

      // calculating the weights
      // 按照论文公式计算每个qexpand_节点增益与最优值weight值。
      for (int nid : qexpand) { // nid - 节点的标号

        bst_node_t parentid = tree[nid].Parent();
        snode_[nid].weight = static_cast<float>(
            evaluator.CalcWeight(parentid, param_, snode_[nid].stats)); // 拆分(split)前 节点的Weight ：
        snode_[nid].root_gain = static_cast<float>(
            evaluator.CalcGain(parentid, param_, snode_[nid].stats)); // 拆分(split)前 节点的Gain ：

      }

    }

    /*! \brief update queue expand add in new leaves */
    // 当每层qexpand_分裂后调用，作为下一层分裂的开始
    // UpdateQueueExpand比较简单，对于qexpand上非叶子节点扩展出左、右节点到newnodes，最后qexpand = newnodes;
    // 个人觉得swap效率更高，没用中间多余的拷贝、释放。所以这个过程类似广度遍历，即xgboost是按照逐层训练的。
    inline void UpdateQueueExpand(const RegTree& tree,
                                  const std::vector<int> &qexpand,
                                  std::vector<int>* p_newnodes) {
      p_newnodes->clear();
      for (int nid : qexpand) {

        if (!tree[ nid ].IsLeaf()) {

          p_newnodes->push_back(tree[nid].LeftChild());
          p_newnodes->push_back(tree[nid].RightChild());

        }
      }
    }

    // update enumeration solution
    //更新枚举最优方案
    inline void UpdateEnumeration(
        int nid, GradientPair gstats, bst_float fvalue, int d_step,
        bst_uint fid, GradStats &c, std::vector<ThreadEntry> &temp, // NOLINT(*)
        TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const {
      // get the statistics of nid
      ThreadEntry &e = temp[nid];
      // test if first hit, this is fine, because we set 0 during init
      if (e.stats.Empty()) {
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      } else {
        // try to find a split
        if (fvalue != e.last_fvalue &&
            e.stats.sum_hess >= param_.min_child_weight) { // e.stats: 左子节点的梯度统计信息
                                                          // 左子节点的二阶梯度的和要满足  >= min_child_weight

          c.SetSubstract(snode_[nid].stats, e.stats);     // c: 右子节点的梯度统计信息 , 
                                                          // 通过 总的 snode_[nid] 减掉左边 e.stats 得到右边 
          if (c.sum_hess >= param_.min_child_weight) { // 右子节点的 二阶梯度的和要满足 >= min_child_weight
            bst_float loss_chg {0};
            if (d_step == -1) { // 反向遍历
              loss_chg = static_cast<bst_float>(
                  evaluator.CalcSplitGain(param_, nid, fid, c, e.stats) -
                  snode_[nid].root_gain);
              bst_float proposed_split = (fvalue + e.last_fvalue) * 0.5f;
              if ( proposed_split == fvalue ) {
                e.best.Update(loss_chg, fid, e.last_fvalue,
                              d_step == -1, c, e.stats);
              } else {
                e.best.Update(loss_chg, fid, proposed_split,
                              d_step == -1, c, e.stats); // 若新的 gain (loss_chg) 比老的 gain 大 则更新最佳切分点
              }
            } else { //前向遍历

              loss_chg = static_cast<bst_float>(
                  evaluator.CalcSplitGain(param_, nid, fid, e.stats, c) -
                  snode_[nid].root_gain);
              bst_float proposed_split = (fvalue + e.last_fvalue) * 0.5f;
              if ( proposed_split == fvalue ) {

                e.best.Update(loss_chg, fid, e.last_fvalue,
                            d_step == -1, e.stats, c);  // 若新的 gain (loss_chg) 比老的 gain 大 则更新最佳切分点
              } else {

                e.best.Update(loss_chg, fid, proposed_split,
                            d_step == -1, e.stats, c);
              }
            }
          }
        }
        // update the statistics
        e.stats.Add(gstats); // 累加梯度值
        e.last_fvalue = fvalue;
      }
    }
    // same as EnumerateSplit, with cacheline prefetch optimization
    // 对于exact greedy算法, 使用缓存预取（cache-aware prefetching）: 对每个线程分配一个连续的buffer，读取梯度信息并存入Buffer中, 然后再统计梯度信息
    // 对于特定特征，枚举出最优分裂值
    void EnumerateSplit(

        const Entry *begin, const Entry *end, int d_step, bst_uint fid,
        const std::vector<GradientPair> &gpair,
        std::vector<ThreadEntry> &temp, // NOLINT(*)
        TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const        
        
// EnumerateSplit基于特征粒度的OMP多线程并行，特征上并行，但依次串行处理qexpand节点，在线程分配到的特定特征子集内，找出对应的最优特征和特征分裂值。
// 使用local cache buffer做缓存预取优化效率，正常逻辑下CSR遍历出特征排序后的实例index是乱序的，
// 访问 position与gpair 不容易构建缓存，因此每次执行特征枚举之前，构建一次cache buffer，预取batch=32的position与gpair到连续内存vector，后续枚举计算容易读取缓存，

        {
      CHECK(param_.cache_opt) << "Support for `cache_opt' is removed in 1.0.0";
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics  清空之前留下的统计信息
      for (auto nid : qexpand) {
        temp[nid].stats = GradStats();
      }
      // right statistics 
      //  GradStats c : 右子节点的统计信息, 包括 HR, GR
      GradStats c;
      // local cache buffer for position and gradient pair
      constexpr int kBuffer = 32; // buffer 的大小
      int buf_position[kBuffer] = {}; // 数组buf_position 的长度为 kBuffer
      GradientPair buf_gpair[kBuffer] = {};
      // aligned ending position
      const Entry *align_end;
      if (d_step > 0) {
        align_end = begin + (end - begin) / kBuffer * kBuffer;
      } else {
        align_end = begin - (begin - end) / kBuffer * kBuffer;
      }
      int i;
      const Entry *it;
      const int align_step = d_step * kBuffer; 

      // internal cached loop 
      for (it = begin; it != align_end; it += align_step) {
        const Entry *p;
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          buf_position[i] = position_[p->index]; // p->index 样本号; position_[i] 第i 样本位于那个节点 
          buf_gpair[i] = gpair[p->index];  // gpair[i] 第i 样本 的梯度信息
        }
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          const int nid = buf_position[i];
          if (nid < 0 || !interaction_constraints_.Query(nid, fid)) { continue; } // 是否 满足了限制条件
          this->UpdateEnumeration(nid, buf_gpair[i],
                                  p->fvalue, d_step,
                                  fid, c, temp, evaluator);
        }
      }

      // finish up the ending piece
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        buf_position[i] = position_[it->index];
        buf_gpair[i] = gpair[it->index];
      }

      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        const int nid = buf_position[i];
        if (nid < 0 || !interaction_constraints_.Query(nid, fid)) { continue; }
        this->UpdateEnumeration(nid, buf_gpair[i],
                                it->fvalue, d_step,
                                fid, c, temp, evaluator);
      }

      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight &&
            c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                evaluator.CalcSplitGain(param_, nid, fid, c, e.stats) -
                snode_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1, c,
                          e.stats);
          } else {
            loss_chg = static_cast<bst_float>(
                evaluator.CalcSplitGain(param_, nid, fid, e.stats, c) -
                snode_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1,
                          e.stats, c);
          }
        }
      }
    }

    // update the solution candidate
    // 更新solution候选集
    virtual void UpdateSolution(const SparsePage &batch,
                                const std::vector<bst_feature_t> &feat_set,
                                const std::vector<GradientPair> &gpair,
                                DMatrix*) {
      // start enumeration
      const auto num_features = static_cast<bst_omp_uint>(feat_set.size());


#if defined(_OPENMP)
      const int batch_size =  // NOLINT
          std::max(static_cast<int>(num_features / this->nthread_ / 32), 1);
#endif  // defined(_OPENMP)
      {
        auto page = batch.GetView();
        dmlc::OMPException exc;
#pragma omp parallel for schedule(dynamic, batch_size)
        for (bst_omp_uint i = 0; i < num_features; ++i) {  // 遍历所有的特征  i-特征的标号

          // -> for( each feature )  // 通过openMP 进行并行处理
          //     -> EnumerateSplit()    // 每个线程执行一个特征 选出对应特征最优的分割值;
          //                            // 在每个线程里汇总各个线程内分配到的数据样本对应的统计量: G(grad) / H(hess)
          //                            // 然后每个线程计算出对应特征下最优分割点;
          exc.Run([&]() { 
            auto evaluator = tree_evaluator_.GetEvaluator();
            bst_feature_t const fid = feat_set[i];
            int32_t const tid = omp_get_thread_num();
            auto c = page[fid]; // page 特征块的集合,  c 特征i 对应的块
            const bool ind =
                c.size() != 0 && c[0].fvalue == c[c.size() - 1].fvalue;
            
            
            // 对于特征存在缺失值情况，会有2次遍历：前向遍历+后向遍历查找，具体算法流程对应论文中的 xgboost稀疏感知算法的前、后遍历流程。
            if (_train_param_.NeedForwardSearch(
                    param_.default_direction, column_densities_[fid], ind)) { // 参数 default_direction 可以控制

               //特征间并行方式
               //每个线程处理 一维特征 fid，遍历数据累计统计量(grad/hess)得到最佳分裂点split_point
              this->EnumerateSplit(c.data(), c.data() + c.size(), +1, fid,
                                  gpair, stemp_[tid], evaluator); // 

            }

            if (_train_param_.NeedBackwardSearch(
                    param_.default_direction) ) {

              this->EnumerateSplit(c.data() + c.size() - 1, c.data() - 1, -1,
                                  fid, gpair, stemp_[tid], evaluator);
            }
          });
        }
        exc.Rethrow();
      }
    }

    // find splits at current level, do split per level
    //逐层分裂，找到 expand节点 的分裂方案
    inline void FindSplit(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          RegTree *p_tree) {
      auto evaluator = tree_evaluator_.GetEvaluator();

      // 每次开始查找前，为了支持随机森林的列采样特性，利用伯努利实现采样feat_index，生成逐层分裂前的特征子集 feat_set
      auto feat_set = column_sampler_.GetFeatureSet(depth);

      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) { //每次调用生成列batch数据，调用UpdateSolution找出batch内的最优特征与特征分裂值。
        this->UpdateSolution(batch, feat_set->HostVector(), gpair, p_fmat);
      }
      
      // after this each thread's stemp will get the best candidates, aggregate results
      // 上面的UpdateSolution() 会为所有待扩展分割的叶结点找到特征
      // 维度的最优分割点，比如对于叶结点A，OpenMP线程1会找到特征f_1
      // 的最优分割点，OpenMP线程2会找到特征f_2的最优分割点, 所以需要
      // 进行全局sync，找到叶结点A的最优分割点。
      this->SyncBestSolution(qexpand); //找到最优解，设为当前分裂节点
      

      // 为需要进行分割的叶结点创建孩子结点, 并计算相应的孩子节点weight 值
      // get the best result, we can synchronize the solution
      for (int nid : qexpand) {
        NodeEntry const &e = snode_[nid];
        
        // now we know the solution in snode[nid], set split
        // 4）对于 qexpand每个节点，
        
        // 若最优增益变化大于阈值 rt_eps，执行树的分裂，分裂成左、右叶子节点;
        // 并设置左、右叶子节点 cright_字段为0，表示该叶子节点是待分裂节点，但是 它们的 cleft_=-1, 任然是叶子节点
        // 重置父节点的 cleft_、cright_，这样父节点本来属于 叶子节点 变成 非叶子节点; 

        // 若最优增益变化 小于 阈值 rt_eps , 不分裂，设置叶子节点的 weight值和 cright_=-1，
        // cright_=-1 , kInvalidNodeId=-1 代表该叶子节点不能够再被分裂，该叶子节点上的所有实例后期会设置 position<0。
        if (e.best.loss_chg > kRtEps) {

          bst_float left_leaf_weight =
              evaluator.CalcWeight(nid, param_, e.best.left_sum) *
              param_.learning_rate;
          bst_float right_leaf_weight =
              evaluator.CalcWeight(nid, param_, e.best.right_sum) *
              param_.learning_rate;
          p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                             e.best.DefaultLeft(), e.weight, left_leaf_weight,
                             right_leaf_weight, e.best.loss_chg,
                             e.stats.sum_hess,
                             e.best.left_sum.GetHess(), e.best.right_sum.GetHess(),
                             0);
        } else {
          (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate); // right  默认为 kInvalidNodeId=-1
        }
      }
    }

    // reset position of each data points after split is created in the tree
    // 每次分裂后更新样本所在的节点信息
    // ResetPosition更新实例的position，即将实例划分到待分裂节点qexpand的候选集上
    inline void ResetPosition(const std::vector<int> &qexpand,
                              DMatrix* p_fmat,
                              const RegTree& tree) {
    

      // set the positions in the nondefault
      this->SetNonDefaultPosition(qexpand, p_fmat, tree);

      // set rest of instances to default position
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid,
      // so that they are ignored in future statistics collection
      
      //2）对特征值为空的数据进行节点划分，由于实例还挂在qexpand节点上，对应节点仍然属于非叶子节点时，会根据默认分裂方向来划分。
//        实际上对于InitData初始删除的实例也会执行该步骤，只不过这部分实例不会被InitNewNode统计，个人觉得没必要对已删除实例进行额外的划分工作。

      const auto ndata = static_cast<bst_omp_uint>(p_fmat->Info().num_row_);

      common::ParallelFor(ndata, [&](bst_omp_uint ridx) { // 遍历所有的 样本数据, 包括特征值为空的 和 特征值不为空的
        CHECK_LT(ridx, position_.size())
            << "ridx exceed bound " << "ridx="<<  ridx << " pos=" << position_.size();
        
        const int nid = this->DecodePosition(ridx);
        if (tree[nid].IsLeaf()) { // 叶子节点(目前没有 左右子节点)
          
          // mark finish when it is not a fresh leaf
          // 做为叶子节点, 并且以后永远也不会分裂了(永远不会有左右子节点),会对处于该叶子节点上的实例 position取反，在将来分裂会被跳过。
          if (tree[nid].RightChild() == -1) { // 
            position_[ridx] = ~nid;
          }

        } else {

          // push to default branch
          if (tree[nid].DefaultLeft()) {
            this->SetEncodePosition(ridx, tree[nid].LeftChild());
          } else {
            this->SetEncodePosition(ridx, tree[nid].RightChild());
          }

        }
      });
    }


    // customization part
    // synchronize the best solution of each node
    // 每个节点 同步 最优方案
    //  3）同步 qexpand节点内每个节点的最优特征与特征分裂值，最优方案是由OMP多线程并行统计，存储在 stemp变量中。
    // 按照 qexpand节点 nid案例遍历下所有线程对应的下 stemp[tid][nid]，调用 NodeEntry方法 best即可。
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        for (int tid = 0; tid < this->nthread_; ++tid) {
          e.best.Update(stemp_[tid][nid].best);
        }
      }
    }

    //设置 特征值非空实例 position
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       DMatrix *p_fmat,
                                       const RegTree &tree) {


      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      
      //首先对特征值非空实例进行节点划分，通过遍历qexpand非叶子节点，生成分裂特征集合fsplits，
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) { // 分批导出数据 , 因为 <SortedCSCPage> 只会存储特征值非空的实例, 因此 batch 中的都是 存储特征值非空的样本
        
        //利用ColIterator获取fsplits的分片数据，进行batch遍历。每个实例通过DecodePosition()获取对应父节点，
        auto page = batch.GetView();

        for (auto fid : fsplits) {
          auto col = page[fid]; // fid - 特征的编号 ; col 特征对应的特征块 
          const auto ndata = static_cast<bst_omp_uint>(col.size());

          common::ParallelFor(ndata, [&](bst_omp_uint j) { // 并发处理 特征块 中的所有行
            const bst_uint ridx = col[j].index; // ridx 样本所在的行id 
            const int nid = this->DecodePosition(ridx); // nid 节点的编号
            const bst_float fvalue = col[j].fvalue; // 这一行的特征值

            // go back to parent, correct those who are not default
            // 根据父节点分裂条件来进入不同子节点，即通过SetEncodePosition()设置实例position()对应的左、右节点上。
            if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
              if (fvalue < tree[nid].SplitCond()) {

                this->SetEncodePosition(ridx, tree[nid].LeftChild());

              } else {

                this->SetEncodePosition(ridx, tree[nid].RightChild());

              }
            }
          });
        }
      }
    }

    // utils to get/set position, with encoded format
    // return decoded position
    // 解码 position
    inline int DecodePosition(bst_uint ridx) const {
      const int pid = position_[ridx];
      return pid < 0 ? ~pid : pid;
    }

    // encode the encoded position value for ridx
    // 编码position
    inline void SetEncodePosition(bst_uint ridx, int nid) {
      if (position_[ridx] < 0) {
        position_[ridx] = ~nid;
      } else {
        position_[ridx] = nid;
      }
    }

    //  --data fields--
    
    //训练参数
    const TrainParam& param_;
    const TrainParam& _train_param_;

    // number of omp thread used during training
    //训练期间的OMP线程数
    const int nthread_;

    common::ColumnSampler column_sampler_;
    
    // Instance Data: current node position in the tree of each instance
    // 每个样本当前节点位置
    std::vector<int> position_;
    
    // PerThread x PerTreeNode: statistics for per thread construction
    // 并行计算，线程间互不影响。每个线程计算的节点分裂信息
    std::vector< std::vector<ThreadEntry> > stemp_; // 二维数组 , 第一维为 线程号 , 第二维为 ThreadEntry
    
    /*! \brief TreeNode Data: statistics for each constructed node */
    //节点统计信息
    std::vector<NodeEntry> snode_;

    /*! \brief queue of nodes to be expanded */
    // 逐层分裂时，对应待分裂节点集合
    std::vector<int> qexpand_;

    TreeEvaluator tree_evaluator_;

    FeatureInteractionConstraintHost interaction_constraints_;
    const std::vector<float> &column_densities_;
  };
};

XGBOOST_REGISTER_TREE_UPDATER(, "grow_")
.describe("Grow tree with parallelization over columns.")
.set_body([]() {
    return new ();
  });
}  // namespace tree
}  // namespace xgboost
