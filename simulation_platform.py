"""
多AI智能体协同的科普创新路径研究 —— 仿真实验平台
==============================================

基于智能体建模(Agent-Based Modeling, ABM)的动态演化仿真系统。
包含基线实验与三个政策干预实验。

智能体类型:
  1. 内容生成智能体 (ContentGenerator)
  2. 知识审核智能体 (KnowledgeReviewer)
  3. 传播分发智能体 (ContentDistributor)
  4. 受众反馈智能体 (AudienceFeedback)

实验设计:
  - 基线实验: 无外部干预的自然演化
  - 实验一: 资源定向投入策略
  - 实验二: 跨域协作激励策略
  - 实验三: 质量反馈闭环策略
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import os

# ============================================================
# 全局设置：支持中文显示
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第一部分：智能体定义
# ============================================================

class AgentType(Enum):
    """智能体类型枚举"""
    CONTENT_GENERATOR = 0    # 内容生成智能体
    KNOWLEDGE_REVIEWER = 1   # 知识审核智能体
    CONTENT_DISTRIBUTOR = 2  # 传播分发智能体
    AUDIENCE_FEEDBACK = 3    # 受众反馈智能体


@dataclass
class Agent:
    """
    智能体基类

    状态变量：
      K: 知识水平 ∈ [0, 1]
      C: 创新能力 ∈ [0, 1]
      Q: 内容质量 ∈ [0, 1]（由K, C, R计算得到）
      R: 声誉/影响力 ∈ [0, 1]
      domain: 所属领域 ∈ {0, 1, ..., num_domains-1}
    """
    agent_id: int
    agent_type: AgentType
    K: float  # 知识水平
    C: float  # 创新能力
    R: float  # 声誉/影响力
    domain: int  # 所属科普领域编号
    Q: float = 0.0  # 内容质量（派生量）

    # 历史记录
    history_K: List[float] = field(default_factory=list)
    history_C: List[float] = field(default_factory=list)
    history_Q: List[float] = field(default_factory=list)
    history_R: List[float] = field(default_factory=list)

    def record(self):
        """记录当前状态到历史"""
        self.history_K.append(self.K)
        self.history_C.append(self.C)
        self.history_Q.append(self.Q)
        self.history_R.append(self.R)


# ============================================================
# 第二部分：仿真环境
# ============================================================

@dataclass
class SimulationConfig:
    """仿真参数配置"""
    # --- 智能体数量 ---
    num_generators: int = 30      # 内容生成智能体数
    num_reviewers: int = 15       # 知识审核智能体数
    num_distributors: int = 20    # 传播分发智能体数
    num_feedback: int = 35        # 受众反馈智能体数
    num_domains: int = 5          # 科普领域数

    # --- 时间参数 ---
    num_steps: int = 200          # 仿真步数
    random_seed: int = 42         # 随机种子

    # --- 知识扩散参数 ---
    alpha_K: float = 0.05         # 知识扩散速率
    eta_K: float = 0.01           # 知识随机涨落强度

    # --- 创新能力参数 ---
    beta_C: float = 0.03          # 创新能力增长速率
    delta_C: float = 0.02         # 创新能力衰减速率

    # --- 内容质量权重 ---
    w_K: float = 0.4              # 知识水平权重
    w_C: float = 0.3              # 创新能力权重
    w_R: float = 0.2              # 声誉权重
    gamma_collab: float = 0.1     # 协作贡献权重

    # --- 声誉更新参数 ---
    lambda_R: float = 0.1         # 声誉更新速率

    # --- 网络参数 ---
    network_rewire_prob: float = 0.05  # 网络重连概率
    edge_formation_base: float = 0.1   # 基础连边概率

    # --- 政策干预参数（实验一：资源定向投入） ---
    resource_boost: float = 0.0        # 资源提升量（基线为0）
    resource_target_type: Optional[AgentType] = None  # 目标智能体类型

    # --- 政策干预参数（实验二：跨域协作激励） ---
    cross_domain_incentive: float = 0.0  # 跨域协作激励系数

    # --- 政策干预参数（实验三：质量反馈闭环） ---
    quality_feedback_enabled: bool = False   # 是否启用质量反馈
    quality_reward: float = 0.0              # 高质量奖励
    quality_penalty: float = 0.0             # 低质量惩罚
    quality_threshold_high: float = 0.7      # 高质量阈值
    quality_threshold_low: float = 0.3       # 低质量阈值


class SimulationEnvironment:
    """
    仿真环境主类

    管理智能体集合、协作网络、动态演化过程。
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.agents: List[Agent] = []
        self.network: nx.Graph = nx.Graph()
        self.step_count = 0

        # 系统级指标历史
        self.history_avg_K = []
        self.history_avg_C = []
        self.history_avg_Q = []
        self.history_avg_R = []
        self.history_innovation_index = []
        self.history_collaboration_density = []
        self.history_gini_quality = []

        self._init_agents()
        self._init_network()

    @property
    def total_agents(self) -> int:
        return (self.config.num_generators + self.config.num_reviewers +
                self.config.num_distributors + self.config.num_feedback)

    def _init_agents(self):
        """初始化智能体群体"""
        agent_id = 0
        type_counts = [
            (AgentType.CONTENT_GENERATOR, self.config.num_generators),
            (AgentType.KNOWLEDGE_REVIEWER, self.config.num_reviewers),
            (AgentType.CONTENT_DISTRIBUTOR, self.config.num_distributors),
            (AgentType.AUDIENCE_FEEDBACK, self.config.num_feedback),
        ]
        for atype, count in type_counts:
            for _ in range(count):
                # 不同类型智能体的初始属性分布不同
                if atype == AgentType.CONTENT_GENERATOR:
                    K0 = self.rng.beta(5, 3)   # 较高知识
                    C0 = self.rng.beta(5, 2)   # 较高创新
                    R0 = self.rng.beta(2, 5)   # 较低声誉（起步）
                elif atype == AgentType.KNOWLEDGE_REVIEWER:
                    K0 = self.rng.beta(7, 2)   # 很高知识
                    C0 = self.rng.beta(2, 5)   # 较低创新
                    R0 = self.rng.beta(3, 3)   # 中等声誉
                elif atype == AgentType.CONTENT_DISTRIBUTOR:
                    K0 = self.rng.beta(3, 4)   # 中等知识
                    C0 = self.rng.beta(3, 3)   # 中等创新
                    R0 = self.rng.beta(4, 3)   # 较高声誉
                else:  # AUDIENCE_FEEDBACK
                    K0 = self.rng.beta(2, 5)   # 较低知识
                    C0 = self.rng.beta(2, 4)   # 较低创新
                    R0 = self.rng.beta(2, 6)   # 较低声誉

                domain = self.rng.randint(0, self.config.num_domains)
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=atype,
                    K=K0, C=C0, R=R0,
                    domain=domain
                )
                self.agents.append(agent)
                self.network.add_node(agent_id)
                agent_id += 1

    def _init_network(self):
        """
        初始化协作网络：小世界网络 + 基于类型的偏好连边

        连边概率公式：
          P(e_ij) = p_base * exp(-|type_i - type_j| / τ) * (1 + sim_domain(i,j))
        其中 sim_domain(i,j) = 1 if domain_i == domain_j else 0
        """
        n = self.total_agents
        tau = 2.0  # 类型差异衰减温度参数

        for i in range(n):
            for j in range(i + 1, n):
                ai, aj = self.agents[i], self.agents[j]
                type_diff = abs(ai.agent_type.value - aj.agent_type.value)
                domain_sim = 1.0 if ai.domain == aj.domain else 0.0
                p = self.config.edge_formation_base * np.exp(-type_diff / tau) * (1.0 + 0.5 * domain_sim)
                if self.rng.random() < p:
                    self.network.add_edge(i, j, weight=self.rng.uniform(0.3, 1.0))

    # ----------------------------------------------------------
    # 核心演化方程
    # ----------------------------------------------------------

    def _compute_diversity_index(self, agent: Agent) -> float:
        """
        计算智能体i的邻域多样性指数 D_i(t)

        D_i(t) = 1 - Σ_k (n_k / |N(i)|)^2    (Herfindahl逆指数)

        其中 n_k 是邻居中类型k的数量, |N(i)| 是邻居总数
        """
        neighbors = list(self.network.neighbors(agent.agent_id))
        if len(neighbors) == 0:
            return 0.0
        type_counts = np.zeros(4)
        for nid in neighbors:
            type_counts[self.agents[nid].agent_type.value] += 1
        proportions = type_counts / len(neighbors)
        hhi = np.sum(proportions ** 2)
        return 1.0 - hhi

    def _compute_collaboration_strength(self, agent: Agent) -> float:
        """
        计算协作强度

        Collab_i(t) = Σ_{j∈N(i)} w_ij * min(K_i, K_j) * (1 + μ * cross_domain_ij)

        其中 cross_domain_ij = 1 if domain_i ≠ domain_j else 0
             μ 为跨域协作激励系数（实验二参数）
        """
        neighbors = list(self.network.neighbors(agent.agent_id))
        if len(neighbors) == 0:
            return 0.0

        collab = 0.0
        for nid in neighbors:
            neighbor = self.agents[nid]
            w_ij = self.network[agent.agent_id][nid].get('weight', 0.5)
            cross_domain = 1.0 if agent.domain != neighbor.domain else 0.0
            incentive = 1.0 + self.config.cross_domain_incentive * cross_domain
            collab += w_ij * min(agent.K, neighbor.K) * incentive
        return collab / len(neighbors)

    def _update_knowledge(self, agent: Agent):
        """
        知识更新方程：

        K_i(t+1) = K_i(t)
                   + α_K * Σ_{j∈N(i)} w_ij * max(K_j(t) - K_i(t), 0)  [知识扩散项]
                   + Δ_R * I(type_i = target)                           [资源注入项]
                   + η_K * ε_i(t)                                       [随机涨落项]
        """
        neighbors = list(self.network.neighbors(agent.agent_id))
        diffusion = 0.0
        if neighbors:
            for nid in neighbors:
                w_ij = self.network[agent.agent_id][nid].get('weight', 0.5)
                diff = max(self.agents[nid].K - agent.K, 0.0)
                diffusion += w_ij * diff
            diffusion /= len(neighbors)

        # 资源注入（实验一）
        resource = 0.0
        if (self.config.resource_target_type is not None and
                agent.agent_type == self.config.resource_target_type):
            resource = self.config.resource_boost

        noise = self.config.eta_K * self.rng.randn()

        agent.K = np.clip(
            agent.K + self.config.alpha_K * diffusion + resource + noise,
            0.0, 1.0
        )

    def _update_creativity(self, agent: Agent):
        """
        创新能力更新方程：

        C_i(t+1) = C_i(t)
                   + β_C * D_i(t) * (1 - C_i(t))       [多样性驱动增长]
                   - δ_C * (1 - D_i(t)) * C_i(t)        [同质化衰减]

        D_i(t) 为邻域多样性指数
        """
        D = self._compute_diversity_index(agent)
        growth = self.config.beta_C * D * (1.0 - agent.C)
        decay = self.config.delta_C * (1.0 - D) * agent.C
        agent.C = np.clip(agent.C + growth - decay, 0.0, 1.0)

    def _update_quality(self, agent: Agent):
        """
        内容质量计算：

        Q_i(t) = w_K * K_i(t) + w_C * C_i(t) + w_R * R_i(t) + γ * Collab_i(t)
        """
        collab = self._compute_collaboration_strength(agent)
        agent.Q = np.clip(
            self.config.w_K * agent.K +
            self.config.w_C * agent.C +
            self.config.w_R * agent.R +
            self.config.gamma_collab * collab,
            0.0, 1.0
        )

    def _update_reputation(self, agent: Agent):
        """
        声誉更新方程：

        R_i(t+1) = (1 - λ) * R_i(t) + λ * Q_i(t) * reach_i(t)

        reach_i(t) = degree_i / max_degree  (归一化度中心性)
        """
        degree = self.network.degree(agent.agent_id)
        max_degree = max(dict(self.network.degree()).values())
        reach = degree / max_degree if max_degree > 0 else 0.0

        agent.R = np.clip(
            (1 - self.config.lambda_R) * agent.R +
            self.config.lambda_R * agent.Q * reach,
            0.0, 1.0
        )

    def _apply_quality_feedback(self, agent: Agent):
        """
        质量反馈闭环机制（实验三）：

        if Q_i(t) ≥ Q_high:
            K_i(t+1) += reward
        elif Q_i(t) ≤ Q_low:
            K_i(t+1) -= penalty
        """
        if not self.config.quality_feedback_enabled:
            return
        if agent.Q >= self.config.quality_threshold_high:
            agent.K = np.clip(agent.K + self.config.quality_reward, 0.0, 1.0)
            agent.C = np.clip(agent.C + self.config.quality_reward * 0.5, 0.0, 1.0)
        elif agent.Q <= self.config.quality_threshold_low:
            agent.K = np.clip(agent.K - self.config.quality_penalty, 0.0, 1.0)

    def _update_network(self):
        """
        网络动态演化：

        1. 以概率 p_rewire 随机断开一条边并重新连接到声誉更高的节点。
        2. 高声誉节点有概率建立新的协作连接（网络扩张）。
        3. 低声誉节点有概率失去连接（网络收缩）。
        
        模拟智能体倾向于与高声誉者协作的偏好，以及协作网络的动态演化。
        """
        edges = list(self.network.edges())
        if not edges:
            return
        
        # 阶段1：网络重连（保持边数不变，但改变连接结构）
        n_rewire = max(1, int(len(edges) * self.config.network_rewire_prob))
        for _ in range(n_rewire):
            if not edges:
                break
            idx = self.rng.randint(0, len(edges))
            u, v = edges[idx]
            # 以声誉差异为依据决定是否断边
            if abs(self.agents[u].R - self.agents[v].R) > 0.3:
                low_r = u if self.agents[u].R < self.agents[v].R else v
                self.network.remove_edge(u, v)
                edges.pop(idx)
                # 低声誉方尝试连接更高声誉的新节点
                candidates = [a for a in self.agents
                              if a.agent_id != low_r and
                              not self.network.has_edge(low_r, a.agent_id)]
                if candidates:
                    # 按声誉加权选择
                    weights = np.array([a.R for a in candidates])
                    weights = weights / (weights.sum() + 1e-10)
                    chosen = self.rng.choice(len(candidates), p=weights)
                    self.network.add_edge(low_r, candidates[chosen].agent_id,
                                          weight=self.rng.uniform(0.3, 1.0))
        
        # 阶段2：网络动态增长/收缩（改变边数，影响密度）
        # 高声誉节点有概率建立新连接（网络扩张）
        avg_reputation = np.mean([a.R for a in self.agents])
        for agent in self.agents:
            # 声誉高于平均的节点有概率新增连接
            if agent.R > avg_reputation and self.rng.random() < 0.02:
                candidates = [a for a in self.agents
                              if a.agent_id != agent.agent_id and
                              not self.network.has_edge(agent.agent_id, a.agent_id)]
                if candidates:
                    # 优先连接不同类型的节点（促进多样性）
                    weights = np.array([
                        1.5 if a.agent_type != agent.agent_type else 1.0 
                        for a in candidates
                    ])
                    weights = weights / weights.sum()
                    chosen = self.rng.choice(len(candidates), p=weights)
                    self.network.add_edge(agent.agent_id, candidates[chosen].agent_id,
                                          weight=self.rng.uniform(0.3, 1.0))
            
            # 声誉很低的节点有概率失去连接（网络收缩）
            if agent.R < 0.2 and self.rng.random() < 0.01:
                neighbors = list(self.network.neighbors(agent.agent_id))
                if len(neighbors) > 1:
                    # 断开与声誉最高邻居的连接（模拟被"抛弃"）
                    max_r_neighbor = max(neighbors, 
                                        key=lambda n: self.agents[n].R)
                    self.network.remove_edge(agent.agent_id, max_r_neighbor)

    # ----------------------------------------------------------
    # 系统级指标计算
    # ----------------------------------------------------------

    def _compute_innovation_index(self) -> float:
        """
        系统创新指数：

        Innovation(t) = (1/N) * Σ_i C_i(t) * D_i(t) * K_i(t)
        """
        total = 0.0
        for agent in self.agents:
            D = self._compute_diversity_index(agent)
            total += agent.C * D * agent.K
        return total / len(self.agents)

    def _compute_gini(self, values: List[float]) -> float:
        """计算基尼系数，衡量质量分布的不平等程度"""
        arr = np.array(sorted(values))
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))

    def _record_system_metrics(self):
        """记录系统级指标"""
        ks = [a.K for a in self.agents]
        cs = [a.C for a in self.agents]
        qs = [a.Q for a in self.agents]
        rs = [a.R for a in self.agents]

        self.history_avg_K.append(np.mean(ks))
        self.history_avg_C.append(np.mean(cs))
        self.history_avg_Q.append(np.mean(qs))
        self.history_avg_R.append(np.mean(rs))
        self.history_innovation_index.append(self._compute_innovation_index())
        self.history_collaboration_density.append(
            nx.density(self.network) if self.network.number_of_nodes() > 1 else 0.0
        )
        self.history_gini_quality.append(self._compute_gini(qs))

    # ----------------------------------------------------------
    # 仿真主循环
    # ----------------------------------------------------------

    def step(self):
        """执行一个仿真步"""
        # 随机打乱更新顺序（异步更新）
        order = self.rng.permutation(len(self.agents))
        for idx in order:
            agent = self.agents[idx]
            self._update_knowledge(agent)
            self._update_creativity(agent)
            self._update_quality(agent)
            self._update_reputation(agent)
            self._apply_quality_feedback(agent)
            agent.record()

        self._update_network()
        self._record_system_metrics()
        self.step_count += 1

    def run(self, verbose: bool = True):
        """运行完整仿真"""
        if verbose:
            print(f"开始仿真: {self.total_agents} 个智能体, {self.config.num_steps} 步")
        for t in range(self.config.num_steps):
            self.step()
            if verbose and (t + 1) % 50 == 0:
                print(f"  步骤 {t+1}/{self.config.num_steps} 完成 | "
                      f"平均质量={self.history_avg_Q[-1]:.4f} | "
                      f"创新指数={self.history_innovation_index[-1]:.4f}")
        if verbose:
            print("仿真完成。")

    def get_results(self) -> Dict:
        """汇总实验结果"""
        return {
            'avg_K': self.history_avg_K,
            'avg_C': self.history_avg_C,
            'avg_Q': self.history_avg_Q,
            'avg_R': self.history_avg_R,
            'innovation_index': self.history_innovation_index,
            'collaboration_density': self.history_collaboration_density,
            'gini_quality': self.history_gini_quality,
            'final_avg_K': self.history_avg_K[-1],
            'final_avg_C': self.history_avg_C[-1],
            'final_avg_Q': self.history_avg_Q[-1],
            'final_innovation': self.history_innovation_index[-1],
            'final_gini': self.history_gini_quality[-1],
        }


# ============================================================
# 第三部分：实验定义
# ============================================================

def create_baseline_config() -> SimulationConfig:
    """基线实验配置：无外部干预"""
    return SimulationConfig(random_seed=42)


def create_experiment1_config() -> SimulationConfig:
    """
    实验一：资源定向投入策略

    对内容生成智能体注入额外知识资源，模拟政府对科普内容创作的定向扶持。
    """
    config = SimulationConfig(random_seed=42)
    config.resource_boost = 0.008          # 每步知识提升量
    config.resource_target_type = AgentType.CONTENT_GENERATOR
    return config


def create_experiment2_config() -> SimulationConfig:
    """
    实验二：跨域协作激励策略

    增加跨领域协作的激励系数，模拟鼓励不同科普领域AI智能体之间的交叉合作。
    """
    config = SimulationConfig(random_seed=42)
    config.cross_domain_incentive = 0.5    # 跨域协作激励系数
    return config


def create_experiment3_config() -> SimulationConfig:
    """
    实验三：质量反馈闭环策略

    引入质量评估反馈机制：高质量智能体获得正向激励，低质量智能体受到约束。
    """
    config = SimulationConfig(random_seed=42)
    config.quality_feedback_enabled = True
    config.quality_reward = 0.005          # 高质量奖励
    config.quality_penalty = 0.003         # 低质量惩罚
    config.quality_threshold_high = 0.6    # 高质量阈值
    config.quality_threshold_low = 0.3     # 低质量阈值
    return config


# ============================================================
# 第四部分：可视化与分析
# ============================================================

class ExperimentVisualizer:
    """实验结果可视化器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_comparison(self, results: Dict[str, Dict], metric: str,
                        ylabel: str, title: str, filename: str):
        """对比多个实验的单一指标"""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        linestyles = ['-', '--', '-.', ':']

        for i, (name, data) in enumerate(results.items()):
            ax.plot(data[metric], label=name,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2)

        ax.set_xlabel('仿真步数', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  图表已保存: {filepath}")

    def plot_all_comparisons(self, results: Dict[str, Dict]):
        """生成所有对比图"""
        metrics = [
            ('avg_Q', '平均内容质量', '各实验平均内容质量演化对比', 'comparison_quality.png'),
            ('avg_K', '平均知识水平', '各实验平均知识水平演化对比', 'comparison_knowledge.png'),
            ('avg_C', '平均创新能力', '各实验平均创新能力演化对比', 'comparison_creativity.png'),
            ('innovation_index', '系统创新指数', '各实验系统创新指数演化对比', 'comparison_innovation.png'),
            ('gini_quality', '质量基尼系数', '各实验质量分布基尼系数演化对比', 'comparison_gini.png'),
            ('collaboration_density', '协作网络密度', '各实验协作网络密度演化对比', 'comparison_density.png'),
        ]
        for metric, ylabel, title, filename in metrics:
            self.plot_comparison(results, metric, ylabel, title, filename)

    def plot_agent_type_analysis(self, env: SimulationEnvironment, exp_name: str):
        """按智能体类型分析各指标"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        type_names = {
            AgentType.CONTENT_GENERATOR: '内容生成',
            AgentType.KNOWLEDGE_REVIEWER: '知识审核',
            AgentType.CONTENT_DISTRIBUTOR: '传播分发',
            AgentType.AUDIENCE_FEEDBACK: '受众反馈',
        }
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

        for attr_idx, (attr, label) in enumerate([
            ('history_K', '知识水平'), ('history_C', '创新能力'),
            ('history_Q', '内容质量'), ('history_R', '声誉影响力')
        ]):
            ax = axes[attr_idx // 2][attr_idx % 2]
            for tidx, (atype, tname) in enumerate(type_names.items()):
                type_agents = [a for a in env.agents if a.agent_type == atype]
                if not type_agents:
                    continue
                all_hist = np.array([getattr(a, attr) for a in type_agents])
                mean_hist = all_hist.mean(axis=0)
                ax.plot(mean_hist, label=tname, color=colors[tidx], linewidth=2)
            ax.set_xlabel('仿真步数', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(f'{label}按智能体类型', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{exp_name} - 各类型智能体指标演化', fontsize=14, fontweight='bold')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'agent_type_{exp_name}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  图表已保存: {filepath}")

    def plot_network_snapshot(self, env: SimulationEnvironment, exp_name: str):
        """绘制终态协作网络"""
        fig, ax = plt.subplots(figsize=(10, 10))
        G = env.network
        type_colors = {
            AgentType.CONTENT_GENERATOR: '#2196F3',
            AgentType.KNOWLEDGE_REVIEWER: '#FF5722',
            AgentType.CONTENT_DISTRIBUTOR: '#4CAF50',
            AgentType.AUDIENCE_FEEDBACK: '#9C27B0',
        }
        node_colors = [type_colors[env.agents[n].agent_type] for n in G.nodes()]
        node_sizes = [300 * env.agents[n].R + 50 for n in G.nodes()]

        pos = nx.spring_layout(G, seed=42, k=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8, ax=ax)

        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=n)
            for atype, (c, n) in zip(
                type_colors.keys(),
                zip(type_colors.values(),
                    ['内容生成', '知识审核', '传播分发', '受众反馈'])
            )
        ]
        ax.legend(handles=legend_elements, fontsize=11, loc='upper left')
        ax.set_title(f'{exp_name} - 终态协作网络', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'network_{exp_name}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  图表已保存: {filepath}")

    def generate_summary_table(self, results: Dict[str, Dict]) -> str:
        """生成结果汇总表"""
        header = f"{'实验名称':<20} {'最终质量':>10} {'最终知识':>10} {'最终创新':>10} {'创新指数':>10} {'基尼系数':>10}"
        lines = [header, '-' * 72]
        for name, data in results.items():
            lines.append(
                f"{name:<20} {data['final_avg_Q']:>10.4f} {data['final_avg_K']:>10.4f} "
                f"{data['final_avg_C']:>10.4f} {data['final_innovation']:>10.4f} "
                f"{data['final_gini']:>10.4f}"
            )
        return '\n'.join(lines)


# ============================================================
# 第五部分：主程序入口
# ============================================================

def run_all_experiments(output_dir: str = None):
    """运行全部实验并生成可视化结果"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_results')

    experiments = {
        '基线实验': create_baseline_config(),
        '实验一(资源定向投入)': create_experiment1_config(),
        '实验二(跨域协作激励)': create_experiment2_config(),
        '实验三(质量反馈闭环)': create_experiment3_config(),
    }

    all_results = {}
    all_envs = {}

    print("=" * 60)
    print("多AI智能体协同科普创新路径 —— 仿真实验平台")
    print("=" * 60)

    for exp_name, config in experiments.items():
        print(f"\n>>> 运行 {exp_name} ...")
        env = SimulationEnvironment(config)
        env.run(verbose=True)
        all_results[exp_name] = env.get_results()
        all_envs[exp_name] = env

    # 可视化
    print("\n>>> 生成可视化结果 ...")
    viz = ExperimentVisualizer(output_dir)
    viz.plot_all_comparisons(all_results)

    for exp_name, env in all_envs.items():
        viz.plot_agent_type_analysis(env, exp_name)
        viz.plot_network_snapshot(env, exp_name)

    # 汇总表
    summary = viz.generate_summary_table(all_results)
    print(f"\n{'=' * 72}")
    print("实验结果汇总")
    print('=' * 72)
    print(summary)

    # 保存数值结果
    numeric_results = {}
    for name, data in all_results.items():
        numeric_results[name] = {
            'final_avg_Q': float(data['final_avg_Q']),
            'final_avg_K': float(data['final_avg_K']),
            'final_avg_C': float(data['final_avg_C']),
            'final_innovation': float(data['final_innovation']),
            'final_gini': float(data['final_gini']),
        }
    results_path = os.path.join(output_dir, 'experiment_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(numeric_results, f, ensure_ascii=False, indent=2)
    print(f"\n数值结果已保存至: {results_path}")

    return all_results, all_envs


if __name__ == '__main__':
    run_all_experiments()
