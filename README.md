## 依赖

* conda 环境

* biopandas

```
conda install -c conda-forge biopandas
```


## 编译python 扩展

```
# 1. install rust environment
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. build python ext
cd rust && maturin develop --release
```