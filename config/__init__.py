"""集中化配置入口（Phase-1 安全引入）。

目标：为分散在各核心模块里的 **路径、文件名、算法超参数** 提供一个统一的
只读入口，避免未来修改时需要多处同步。

当前阶段（Phase 1）：本包**不会**改变任何现有模块的行为——它只是一个**可选的**
聚合层，把真相来源从各模块里重新导出到这里，外部工具（如
``tools/status_board.py``）和新代码可以直接 ``from config import paths``
或 ``from config import constants`` 使用。

后续阶段（Phase 3）：把各模块里的硬编码常量定义**迁到** config/，让
原定义改为从这里导入。那一步是破坏性的，需要逐模块测试，不在本阶段做。

使用约定：
- ``config.paths``：所有相对于项目根的目录名 / 文件名。
- ``config.constants``：算法超参数、维度、动作空间等全局常量。
- 两个子模块都只导出常量，不包含任何副作用代码。
"""

from . import paths  # noqa: F401
from . import constants  # noqa: F401

__all__ = ["paths", "constants"]
