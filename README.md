Shared Latent Space for Diverse OpenVSP Aircraft Configurations

-------------------------
1. Clone the repository
-------------------------

Use your preferred method to clone the repo:

    git clone <your-repo-url>
    cd <your-repo-folder>

-------------------------------
2. Install required packages
-------------------------------

Make sure you have Python and pip installed.

Install dependencies with:

    pip install -r req.txt

------------------------
3. Train the model
------------------------

Run:

    python vae_train.py

Steps:
a. Input data should be placed in:
    - .obj files  --> data_obj/
    - .binvox     --> data_vox/

b. Choose the loss function inside vae_train.py
   Or define a custom one in Loss/loss_general.py

c. The following hyperparameters need to be changed in code:
    - BATCH_SIZE
    - LEARNING_RATE
    - HIDDEN_DIM

d. If voxelization is needed:
    - Run the script normally
    - Choose a resolution
    - Answer "yes" to the question: Voxelize?

e. If reconstruction is needed:
    - Type "y" when prompted
    - Reconstructions are saved in:
        recostruction_latent/vae_{loss_name}.pth

f. Trained model is saved in:
        Models/

--------------------------------------------
4. Visualize latent space (3D projection)
--------------------------------------------

Run:

    python visualize_latent.py

a. Make sure the hyperparameters match those used in training  
b. Select the correct model path in the code

---------------------------------------------
5. Casual (type-unknown) generation
---------------------------------------------

Run:

    python casual_generation.py

a. Set correct hyperparameters in the code  
b. Choose the model to use (edit code)  
c. Generated designs will be saved in:
       casual_design/

---------------------
6. Filtering
---------------------

Run:

    python filtering.py

a. In the script, choose the folder with designs to filter  
b. Set a threshold (manually in code)

------------------------------------------------
7. Non-Casual (type-known) guided generation
------------------------------------------------

Run:

    python non-casual_generation.py

a. Set the correct hyperparameters in the code  
b. Choose a design to be “changed”  
c. Set how many variants to generate  
d. Add path of the model to use  
e. Generated files will be saved in:
       guided_generation/

------------------------------------------------

For any issues or questions, feel free to contact the author or open an issue.


Binvox and viewvox:  P.Min,Binvox, http://www.patrickmin.com/binvox, Accessed: 2025-04-15, 2004- 2019.
