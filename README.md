# Subject
abstract

## Overview
- Objective :
- Method :
- Tools : 도구A, 도구B, 도구C

## Features
- PDF 자료 기반 정보 추출 (예: 자료, 기사 등)
- ...
- 확증 편향 방지 전략 : ....

## Tech Stack

| Category   | Details                      |
|------------|------------------------------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini via OpenAI API   |
| Retrieval  | FAISS(Hit Rate@K, MRR)       |
| Embedding  | multilingual-e5-large        |

## Agents

- Agent A: Assesses technical competitiveness
- Agent B: Evaluates market opportunity and team capability

## Architecture
(그래프 이미지)

## Directory Structure
├── data/                  # PDF 문서
├── agents/                # Agent 모듈
├── prompts/               # 프롬프트 템플릿
├── outputs/               # 평가 결과 저장
├── app.py                 # 실행 스크립트
└── README.md

## Contributors
- 이한결 : Prompt Engineering, Agent Design
- 한채윤 : PDF Parsing, Retrieval Agent