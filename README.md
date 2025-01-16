# Machine Learning Engineering Challenge

## Summary of the Challenge
This challenge involves preparing a Machine Learning model for deployment, ensuring it can handle predictions on request. The task includes creating all necessary files, documentation, and code to make the model production-ready while allowing flexibility for improvements in the pipeline. The solution should work locally, with the possibility of being adapted for any cloud provider.

## Description
**Requirements:**
1. Deploy the `model.py` file provided by teammate Mike to production.
2. Ensure the model can receive and predict data upon request.
3. The solution should work for any cloud provider.
4. Consider this as version 1 of a series of future model versions.
5. Optional: Propose and implement a better model pipeline if the provided one is unsuitable.

**Guidance:**
- Add files, code, and documentation to improve the repository.
- Ensure the solution is functional locally; cloud deployment is not required.
- Provide reasoning for decisions made during the process.
- Highlight assumptions or questions for the team in the documentation.
- Consider the model's binary classification use case with imputation requirements.

## Solution Proposed
**Workflow**
Composed of:
- Training Pipeline (generate imputation and model dependencies)
- Inference Pipeline (real-time prediction through an API request)
- Streamlit (Interface to request an inference)

The requirement '*The solution should work for any cloud provider*' is satisfied by the Docker implementation (cloud provider agnostic).

**Next Steps**
- Batch Inference Pipeline (the workflow is only prepared for one prediction at time)
- ML Improvements (Feature Engineering, MICE Imputation, Model Evaluation)

## ML Improvements
1. **Data Validation:** Added validation checks for input data to ensure robustness.
2. **Improved Imputation:** 
   - Replaced constant imputation with mean imputation, given the fact that `Age == 0` is odd.
        - Mean is not ideal, but it is an improvement over constant imputation.
3. **Model Upgrade:** 
   - Switched from `LogisticRegression` to `RandomForestClassifier`. Avoided the dependency on StandardScaler required by LogisticRegression.
4. **Model Evaluation:** Implemented 5-Fold Stratified Cross-Validation (without data leakage) to ensure robust metrics during training, replacing a simple hold-out approach.
5. **Complete Training:** Finalized the training process after thorough model validation.

## Install Dependencies
1. Navigate to the repository folder:
   ```bash
   cd <repository_folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

3. Install the project in editable mode (`setup.py`):
   ```bash
   pip install -e .
   ```


## How to Execute

### Training pipeline
1. **Run using terminal:**
```bash
python run app/train.py
```

### Inference pipeline
You have two options to deploy your inference API:
1. **Run using terminal:**
```bash
python run app/inference.py
```

2. **Execute a container (build your image and run your container)**
```bash
docker build -t mle-challenge-deploy .
```
```bash
docker run --name inference-app -p 8089:8089 mle-challenge-deploy
```
### Streamlit Application
1. **Run using terminal:**
```bash
streamlit run app/main.py
```

## Docker Setup (Windows 10)
To install Docker Desktop on your Windows machine, is required to:
- Turn On `Virtualization` on your computer BIOS
- Install the following features on your System: `HyperV` and `Virtual Machine Platform`
- Install `WSL 2`
- Install `Docker Desktop`
To be successful, your Docker Desktop should have your docker engine running (if stopped, you failed something).

For more details, follow the [Docker Documentation Guide](https://docs.docker.com/desktop/setup/install/windows-install/).
