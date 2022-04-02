# 環境チーム
# from env_team import Evaluator
from wm_team import WorldModel
from algo_team.DIAYN import DIAYN
from algo_team.Trainer import Trainer
from algo_team.config.LL import config

# 世界モデルチーム
import gym
env = gym.make("LunarLanderContinuous-v2")
worldmodel = WorldModel(env)
worldmodel.train()
# アルゴリズムチーム
config.low_env = worldmodel
config.high_env = worldmodel

diayn = Trainer(config, DIAYN)
diayn.train()

# 環境チーム
# ev = Evaluator(env, diayn)  # diayn の性能を評価する。元の環境に依存する？状態の網羅性や報酬などを見るべきか
# ev.evaluate(diayn)
