  
// Instance Data: current node position in the tree of each instance
// 每个样本当前节点位置 eg. position_[1]=0 : 1号 样本位于 0号节点
std::vector<int> position_;

// PerThread x PerTreeNode: statistics for per thread construction
// 并行计算，线程间互不影响。每个线程计算的节点分裂信息
std::vector< std::vector<ThreadEntry> > stemp_;

/*! \brief TreeNode Data: statistics for each constructed node */
//节点统计信息
std::vector<NodeEntry> snode_;

/*! \brief queue of nodes to be expanded */
// 逐层分裂时，对应待分裂节点集合
std::vector<int> qexpand_;


/*! \brief Element from a sparse vector */
struct Entry {
  /*! \brief feature index 样本的ID  */
  bst_feature_t index;
  /*! \brief feature value */
  bst_float fvalue;
  /*! \brief default constructor */
  Entry() = default;


  /*! \brief core statistics used for tree construction */
struct XGBOOST_ALIGNAS(16) GradStats {
  using GradType = double;
  /*! \brief sum gradient statistics  一阶梯度的和 G */
  GradType sum_grad { 0 };
  /*! \brief sum hessian statistics  二阶梯度的和 H */
  GradType sum_hess { 0 };
  
  
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


  struct SplitEntryContainer {
  
  /*! \brief loss change after split this node */
   // 节点分裂的增益loss变化值
  bst_float loss_chg {0.0f};
  
  /*! \brief split index */
   // 分裂特征的index
  bst_feature_t sindex{0};
  
  // 特征的分裂值
  bst_float split_value{0.0f};

  GradientT left_sum;
  GradientT right_sum;

  SplitEntryContainer() = default;