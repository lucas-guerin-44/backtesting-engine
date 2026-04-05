include .env
export

backend:
	uvicorn api:app --reload --port $(LOCAL_API_PORT)

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C

frontend:
	set DATALAKE_URL=$(DATALAKE_URL) && streamlit run frontend.py
else
frontend:
	DATALAKE_URL=$(DATALAKE_URL) streamlit run frontend.py
endif