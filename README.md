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

## python 开发手册

### 使用 `vscode` 作为`ide`

vscode 建议配置如下

```json
{
    "editor.formatOnType": true,
    "editor.formatOnSave": true,
    "python.pythonPath": "~/anaconda3/bin/python",
    "terminal.integrated.scrollback": 100000,
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.mypyEnabled": false,
    "python.linting.pycodestyleEnabled": false,
    "python.formatting.provider": "black",
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

* 所需插件
    * `python vscode`插件

### 代码格式, lint相关

* `python`代码格式化使用`black`
* `python lint`使用`flake8`
* `python`代码使用`isort`自动排序`import`
