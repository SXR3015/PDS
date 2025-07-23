# TriRef-DualDiff3D
## Environment Setup

First, install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the Model
To run the model, follow these steps:

**Two-Stage Training**

1.Train the Dual-Modal Diffusion Model

 * Open `opt.py` and set <mark> opt.refine </mark>  = False.
 * Run the ` main.py ` to train the diffusion model.
   
2.Load the Trained Model

* After training completes, reset <mark> opt.resume_path </mark>  to the path of your best-performing model weights.
* Ensure <mark> opt.refine </mark>  = True is still set.


3.To save intermediate images during training
* Adjust the saving steps parameter in either ` train.py ` or ` train_refine.py `.

4.Hyperparameter Configuration

* All other hyperparameters can be configured in ` opt.py `.
* Image size need to be adjusted by the input size
