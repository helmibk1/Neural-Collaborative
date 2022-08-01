import torch
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint

class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 20,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'device_id': 0,
                'use_cuda': True,
                'pretrain': True,
                'pretrain_mf': 'C:/Users/helmi/Desktop/neural-collaborative-filtering-master/neural-collaborative-filtering-master/src/checkpoints/gmf/{}'.format('gmf_factor8neg4-implict_Epoch1_HR0.1033_NDCG0.0465.model'),
                'pretrain_mlp': 'C:/Users/helmi/Desktop/neural-collaborative-filtering-master/neural-collaborative-filtering-master/src/checkpoints/{}'.format('mlpmlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_Epoch0_HR0.4502_NDCG0.2512.model'),
                'model_dir':'C:/Users/helmi/Desktop/neural-collaborative-filtering-master/neural-collaborative-filtering-master/src/checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

path = 'C:/Users/helmi/Desktop/neural-collaborative-filtering-master/neural-collaborative-filtering-master/src/checkpoints/{}'.format('pretrain_neumf_factor8neg4_Epoch0_HR0.4368_NDCG0.2163.model')
importedModel = NeuMF(neumf_config)
def resume_checkpoint(model,config,model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)

resume_checkpoint(importedModel,neumf_config,path,0)

print(importedModel)
def predict(user,movie):
    importedModel.eval()
    score = importedModel(user, movie)
    print(score)
user = torch.tensor([[1., -1.], [1., -1.]]).to(torch.int64)
movie = torch.tensor([[1., -1.], [1., -1.]]).to(torch.int64)
predict(user,movie)

