from algo_team.config.LL import config
from algo_team.Trainer import Trainer
from algo_team.DIAYN import DIAYN

trainer = Trainer(config,DIAYN)
trainer.train()