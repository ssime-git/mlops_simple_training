setup:
	pip install -r requirements.txt

get-modal-token:
	modal token new

create-wandb-secret-in-modal:
	modal secret create my-wandb-secret WANDB_API_KEY=<wandb-token>

deploy-modal:
	modal deploy modal_train.py

run-server:
	uvicorn app:app --reload
	
trigger-training:
	curl -X POST http://localhost:8000/trigger-training