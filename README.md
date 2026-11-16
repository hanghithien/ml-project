## Create Environment
### Option 1: Venv
```bash
python -m venv venv
venv\Scripts\activate
```

### Option 2: Conda
```bash
conda create -n myenv python=3.13
```

## Requirement
`Python:` 3.13
`pip install -r requirements.txt`

## How to run
Always run commands from the project root directory and using the `-m` flag to avoid import path errors for the src directory.
Ex:

```
python -m src.preprocess
python -m src.model_b
python -m src.model_c
```

## Goodluck !
![alt text](image.png)


