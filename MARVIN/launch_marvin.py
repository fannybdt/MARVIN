from marvin import MARVINPipeline
M = 32
K = 14
D = 32

batch_size = 64
eval_every = 500
num_epochs = 60
save_every = 20000
root = "scyan_datasets/aml/aml_true.csv"
mymodel = MARVINPipeline(M, K, D, batch_size, eval_every, num_epochs, root, save_every, 
                  factor = 8, lr = 2*1e-3, dropout = 0.3, ID = "AML", ratio_supervision=0.5)
mymodel.train()
mymodel.save(f"{mymodel.directory}/MARVIN_{mymodel.ID}_final.pt")

