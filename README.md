# Parallel Machine Scheduling: Time-Indexed Formulation & Column Generation

[![Python](https://img.shields.io/badge/Python-2.7%20%7C%203.x-blue.svg)](https://www.python.org/)
[![Solver](https://img.shields.io/badge/Solver-Gurobi-orange.svg)](https://www.gurobi.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Machine__Scheduling__Problems-181717?logo=github)](https://github.com/fabio-emanuel/Machine_Scheduling_Problems)

Implementação e análise comparativa de abordagens exatas e heurísticas via **Decomposição de Dantzig-Wolfe (Geração de Colunas)** e **Programação Inteira Mista (MIP)** para o problema de escalonamento em máquinas paralelas idênticas com minimização de atraso ponderado ($P_m \mid r_j \mid \sum w_j T_j$).

O trabalho estende a metodologia desenvolvida no artigo de referência:
> **Van den Akker, J. M., Hurkens, C. A. J., & Savelsbergh, M. W. P. (2000).** *Time-Indexed Formulations for Machine Scheduling Problems: Column Generation*. INFORMS Journal on Computing, 12(2), 111-124.

---

## 📌 Visão Geral do Problema

As formulações indexadas no tempo (*time-indexed*) fornecem relaxações lineares muito consistentes para problemas de agendamento de tarefas. Contudo, seu tamanho cresce de forma acentuada com o aumento do horizonte de tempo ($T$) e do número de tarefas ($n$).

Neste projeto, o modelo original de máquina simples é estendido para **múltiplas máquinas paralelas idênticas ($K$ máquinas)**, buscando minimizar o atraso ponderado:
$$\min \sum_{j=1}^{J} \sum_{t=1}^{T - p_j + 1} \sum_{k=1}^{K} c_{jt} \, x_{jtk}$$

---

## 🏗️ Estratégias de Solução

### 1. Modelo MIP Compacto Exato (`exato.py`)
- Formulação direta com variáveis binárias $X_{j,t,k} \in \{0, 1\}$.
- Executado via **Gurobi Optimizer**.
- Garante otimalidade inteira em instâncias pequenas, mas atinge limites práticos de memória/tempo em problemas de escala industrial.

### 2. Decomposição de Dantzig-Wolfe / Geração de Colunas (`pmp_novo4_heuristica.py`)
- **Heurística Inicial (Warm-start):** Algoritmo *Random Multi-Start* baseado no índice de prioridade $d_j / w_j$, fornecendo uma base inicial de colunas viáveis.
- **Problema Mestre Restrito (RMP):** Trabalha com variáveis de pesos de programas de produção ($\lambda$), assegurando a atribuição única por tarefa e a restrição de convexidade global para $K$ máquinas.
- **Subproblema de Precificação (Pricing):** Modela a identificação de novos planos de produção com custo reduzido negativo como a busca de **Caminho Mínimo em Grafo Acíclico Direcionado (DAG)**, resolvido em $O(nT)$ via Programação Dinâmica.
- **Separação por Máquinas:** Resolução de modelo LP auxiliar para distribuição das frações de tarefas entre as máquinas físicas.

---

## 📊 Desempenho Computacional

| Instância | Abordagem | Tempo de Processamento | Função Objetivo (LP / MIP) |
| :--- | :--- | :--- | :--- |
| **20 Tarefas (3 Mqs)** | Gurobi Exato (MIP) | **1.45 s** | 279.00 (Ótimo Inteiro) |
| | Relaxação Linear (MIP) | 0.10 s | 269.62 (Relaxado) |
| | Geração de Colunas (D-W) | 0.35 s | 269.62 (Relaxado) |
| **200 Tarefas (5 Mqs)** | Gurobi Exato (MIP) | > 68 min (GAP 3.91%) | 7153.00 (Incumbent) |
| | Relaxação Linear (MIP) | 798.80 s | 6873.36 (Relaxado) |
| | Geração de Colunas (D-W) | **164.56 s** (~5x mais rápido) | 6873.36 (Relaxado) |

### 📈 Convergência e Parada Prematura (*Tailing-off*)
- A cada iteração, é calculado o *Lower Bound* Lagrangiano ($z_{LP} \ge \sum \pi_j + \min \bar{c}_k + \alpha$).
- Na instância de 200 tarefas, interromper o algoritmo ao atingir um GAP de **1%** economizou cerca de **30% do tempo total** de execução, preservando uma cota inferior de alta precisão.

---

## 📁 Estrutura do Repositório

```text
├── exato.py                    # Modelo exato compacto (Gurobi MIP)
├── pmp_novo4_heuristica.py     # Algoritmo de Geração de Colunas (D-W + Programação Dinâmica)
├── inputs.txt                  # Arquivo com parâmetros da instância (T, M, tarefas, p, d, w)
├── Projecto_vf(2).pptx         # Slides com detalhamento matemático e gráficos de convergência
└── README.md                   # Documentação do projeto
