
## xgboost 源码解析

仅保留主流程的源码, 并对其进行必要的注释

1.mian 函数入口

src/cli_mian.cc

2.划分树节点的精确贪心算法

src/tree/updater_colmaker.cc

3.划分树节点的近似算法

src/tree/updater_histmaker.cc

4.数据的稀疏存储 Dmatrix 

src/data/data.cc

可以结合博客和本项目的源码注释理解 xgboost 的原理

博客地址

[1] https://www.yinxiang.com/everhub/note/b5ffa121-50f1-4315-aa7b-3e84bff90e47

[2] https://www.yinxiang.com/everhub/note/f58ddd32-91ea-40e2-bc8a-1c44a615ba9a

[3] https://www.yinxiang.com/everhub/note/14860d93-5ce9-43a5-8f3e-590db6fc3f58