from pathlib import Path
from useb import run
import torch
import pickle
from models import VICReg
from argparse import ArgumentParser
from transformers import BertTokenizerFast
import pdb

class EvalModel(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.model = VICReg(args)           
        self.model.eval()   
    
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', padding=True,truncation=True)
        
    @torch.no_grad()
    def forward(self, x):
        
        model_input = self.tokenizer(x,return_tensors='pt',return_token_type_ids=False,truncation=True,padding='max_length',max_length=self.model.args.seq_len)
        
        for k,v in model_input.items():
            model_input[k] = v.cuda()
        embedding = self.model(model_input,only_inference=True)
            
        return embedding

def eval(args):   

    # epoch = args.ckpt.parts[-1].split('-')[0].split('=')[1]
    step = args.ckpt.stem.split("_")[-1]
    print(f"\nEvaluating checkpoint with tags {args.ckpt.parts[-2].split('_')[1:]}, at step : {step}\n")
    # else:
    #     print(f"\nEvaluating barlow bert at {args.ckpt}")
    evalmodel = EvalModel(args).cuda()
        
    # The only thing needed for the evaluation: a function mapping a list of sentences into a batch of vectors (torch.Tensor)
    @torch.no_grad()
    def semb_fn(sentences) -> torch.Tensor:
        return evalmodel(sentences)
    
    results, results_main_metric = run(
                    semb_fn_askubuntu=semb_fn, 
                    semb_fn_cqadupstack=semb_fn,  
                    semb_fn_twitterpara=semb_fn, 
                    semb_fn_scidocs=semb_fn,
                    eval_type='valid',
                    data_eval_path='data-eval'  # This should be the path to the folder of data-eval
    )
    
    # if args.model=='barlow':
    #     outfile = '/'.join([args.model]+list(args.ckpt.parts[-2:]))
    # else:
    #     outfile = args.ckpt.with_suffix('.pkl')
    outfile = Path('/'.join(['results','vicreg']+list(args.ckpt.parts[-2:]))).with_suffix('.pkl')    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    if outfile.exists():
        print(f'Output file {outfile} already exists. Please save results manually.')
        pdb.set_trace()
    pickle.dump((results,results_main_metric),open(outfile,'wb'))

if __name__=="__main__":
    
    parser = ArgumentParser(description='TSDAE Evaluation')
    parser.add_argument("--arch", type=str, default="bert-base-uncased",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="1024",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=100.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=5.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--use-param-weights", action='store_true', 
                        help='Use learnable weights for the three losses')
    parser.add_argument("--seq_len", type=int, default=128,
                        help='Sequence length for Transformer model')
    parser.add_argument('--ckpt', type=Path)
    vicreg_ckpt = "--ckpt /mounts/Users/cisintern/jabbar/data/vicreg/bookcorpus_mlp1024_bs384_lr1e3_gpu8_epoch1_warmup0.2_sim25_std100_cov5_run2/ckpt_step_115632.pth"
    tmp_args = vicreg_ckpt.split()
    args = parser.parse_args()
    
    eval(args)