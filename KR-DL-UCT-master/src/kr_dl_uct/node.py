"""
这段代码实现了一个基于树状结构的 Node 类，类似于用于强化学习或搜索算法的 MCTS（蒙特卡洛树搜索）。以下是各部分的含义：
Node 类:
    m_v: 存储节点的评估值（value）。
    m_visits: 存储节点的访问次数。
    m_move: 表示节点对应的动作。
    m_prob: 节点的概率，用于决策树搜索。
    m_children: 子节点的列表。
    m_parent: 父节点的引用。
核心方法:
    add_node: 添加子节点。
    ucb_select: 基于 UCB 算法从子节点中选择最佳节点。
    sort_children: 按访问次数和评估值对子节点排序。
    get_eval: 根据颜色返回当前节点的评估值。
    compute_eval: 静态方法，用于计算评估值。
辅助类 NodeComp:
    实现了一个比较器，用于比较两个节点。首先比较访问次数，其次比较评估值。

"""

from typing import List, Optional, Tuple

class KDE:
    """Placeholder for KDE class used in the original code."""
    pass

class Node:
    def __init__(self, x: float, y: float, spin: float, prob: float):
        self.m_v = 0.0  # Value estimate
        self.m_visits = 0.0  # Visit count
        self.m_move = [x, y, spin]  # Move representation
        self.m_prob = prob  # Probability of this node
        self.m_children: List[Node] = []  # Children nodes
        self.m_parent: Optional[Node] = None  # Parent node
        self.m_kde_0 = None  # KDE instance for evaluation
        self.m_kde_1 = None  # KDE instance for evaluation
        self.m_init_infos: List[Tuple[int, float]] = []  # Additional initialization info

    def sort_children(self, is_white: bool):
        """Sort children based on visits and evaluation value."""
        self.m_children.sort(
            key=lambda child: (child.m_visits, child.get_eval(is_white)),
            reverse=True
        )

    def ucb_select(self, is_white: bool, ucb_const: float) -> Optional['Node']:
        """Select child using Upper Confidence Bound (UCB)."""
        if not self.m_children:
            return None
        return max(
            self.m_children,
            key=lambda child: (
                child.get_eval(is_white) +
                ucb_const * (child.m_prob / (1 + child.m_visits))
            )
        )

    def add_node(self, x: float, y: float, spin: float, prob: float) -> 'Node':
        """Add a new child node."""
        new_node = Node(x, y, spin, prob)
        new_node.m_parent = self
        self.m_children.append(new_node)
        return new_node

    def add_init_info(self, id_: int, prob: float):
        """Add initialization info."""
        self.m_init_infos.append((id_, prob))

    def is_root(self) -> bool:
        """Check if this node is the root."""
        return self.m_parent is None

    def is_first_update(self) -> bool:
        """Check if this is the first update."""
        return self.m_visits == 0

    def has_children(self) -> bool:
        """Check if the node has children."""
        return len(self.m_children) > 0

    def sample_move(self, num_sample: int, l: float) -> List[float]:
        """Sample moves for exploration (placeholder)."""
        return [0.0] * num_sample  # Placeholder implementation

    def get_children(self) -> List['Node']:
        """Get the list of children."""
        return self.m_children

    def get_num_children(self) -> int:
        """Get the number of children."""
        return len(self.m_children)

    def get_parent(self) -> Optional['Node']:
        """Get the parent node."""
        return self.m_parent

    def get_move(self) -> List[float]:
        """Get the move associated with this node."""
        return self.m_move

    def get_init_info(self, nth: int) -> Tuple[int, float]:
        """Get initialization info."""
        return self.m_init_infos[nth]

    def get_visits(self) -> float:
        """Get the number of visits."""
        return self.m_visits

    def get_prob(self) -> float:
        """Get the probability."""
        return self.m_prob

    def get_exp_v(self, is_white: bool) -> float:
        """Get expected value."""
        return self.m_v if is_white else -self.m_v

    def get_eval(self, is_white: bool) -> float:
        """Get evaluation value."""
        return self.get_exp_v(is_white)

    @staticmethod
    def compute_eval(dist_v: List[float], is_white: bool) -> float:
        """Compute evaluation value based on distribution vector."""
        return sum(dist_v) if is_white else -sum(dist_v)

    def kr_update(self, v: float):
        """Update kernel regression (placeholder)."""
        self.m_v = v


class NodeComp:
    def __init__(self, is_white: bool):
        self.m_is_white = is_white

    def __call__(self, a: Node, b: Node) -> bool:
        """Compare two nodes for sorting."""
        if a.get_visits() != b.get_visits():
            return a.get_visits() < b.get_visits()
        return a.get_eval(self.m_is_white) < b.get_eval(self.m_is_white)