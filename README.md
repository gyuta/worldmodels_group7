# 世界モデルと知能 グループ7

## 目標

* Learning World Models with Skills
  * 世界モデルではデータから外界の予測モデルを獲得するが，そのためのデータをどのように取得するのかが問題になる．ここでは，(1)DIAYN[^1]やDADS[^2]に基づくスキルの獲得，(2) 獲得したスキルに基づいた探索によるデータの取得と世界モデルの学習を交互に繰り返すことの有効性を検証する．Mujoco，Habitat[^3]，Soft Gym[^4]などの環境で有効性を評価する．

## ipynb

* [最終イメージ.ipynb](https://colab.research.google.com/drive/1G-2ubL8gU18NEBjQo1NHRVQmfXTjRPnV?usp=sharing)

## 参考文献

* [スキルに基づく探索方策による世界モデルの学習](https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_2J4GS8c05/_pdf/-char/ja)
* [強化学習における報酬なしスキル獲得の階層化](https://www.jstage.jst.go.jp/article/pjsai/JSAI2019/0/JSAI2019_4Rin103/_pdf/-char/ja)
* [Diversity is All You Need: Learning Diverse Skills without a Reward Function](https://sites.google.com/view/diayn/)
* [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070)
* [Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657)
* [Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405)
* [SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation](https://arxiv.org/abs/2011.07215)

## GitHub

* [Deep Reinforcement Learning Algorithms with PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
* [Soft Actor-Critic, DIAYN.md](https://github.com/ben-eysenbach/sac/blob/master/DIAYN.md)
* [Diversity Is All You Need (DIAYN) Implementation using RLkit](https://github.com/johnlime/RlkitExtension)
* [Exploration through Hierarchical Meta Reinforcement Learning](https://github.com/navneet-nmk/Hierarchical-Meta-Reinforcement-Learning)

[^1]: [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070)
[^2]: [Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657)
[^3]: [Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405)
[^4]: [SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation](https://arxiv.org/abs/2011.07215)
