# Classificador de Dígitos MNIST com PyTorch

Este projeto implementa uma Rede Neural Artificial do tipo Multi-Layer Perceptron (MLP) utilizando a biblioteca PyTorch para classificar imagens de dígitos manuscritos do dataset MNIST.

## Visão Geral

O código, originalmente apresentado em um formato de Jupyter Notebook, realiza as seguintes etapas:

1.  **Importação de Bibliotecas:** Importa as bibliotecas necessárias (PyTorch, NumPy, Matplotlib, etc.).
2.  **Carregamento e Preparação dos Dados:**
    * Baixa o dataset MNIST (se não estiver presente localmente).
    * Aplica transformações básicas (converte imagens para tensores PyTorch).
    * Cria DataLoaders para carregar os dados em lotes (batches) para treino (`trainloader`) e validação (`valloader`).
3.  **Visualização:** Mostra um exemplo de imagem do dataset.
4.  **Definição do Modelo:**
    * Define uma classe `Modelo` que herda de `torch.nn.Module`.
    * A arquitetura consiste em:
        * Uma camada de entrada linear que achata a imagem 28x28 (784 neurônios) para 128 neurônios.
        * Uma camada oculta linear de 128 para 64 neurônios.
        * Uma camada de saída linear de 64 para 10 neurônios (correspondendo aos 10 dígitos, 0-9).
    * Utiliza a função de ativação ReLU após as camadas ocultas.
    * Aplica `LogSoftmax` na saída final, adequada para uso com a função de perda `NLLLoss`.
5.  **Função de Treinamento (`treino`):**
    * Define o otimizador (SGD com momento) e a função de perda (NLLLoss).
    * Itera sobre o dataset de treino por um número definido de épocas (EPOCHS = 10).
    * Em cada época, itera sobre os lotes de dados.
    * Para cada lote:
        * Realiza a passagem para frente (forward pass) pelo modelo.
        * Calcula a perda (`perda_instantanea`).
        * Realiza a retropropagação (backpropagation) para calcular os gradientes.
        * Atualiza os pesos do modelo usando o otimizador.
    * Imprime a perda média da época e o tempo total de treino.
6.  **Função de Validação (`validacao`):**
    * Avalia o modelo treinado no dataset de validação.
    * Itera sobre os dados de validação sem calcular gradientes (`torch.no_grad()`) para economizar memória e acelerar o processo.
    * Para cada imagem:
        * Obtém as probabilidades preditas pelo modelo.
        * Determina a classe predita (`etiqueta_pred`) como a de maior probabilidade.
        * Compara com a classe real (`etiqueta_certa`).
        * Calcula a precisão geral do modelo.
    * Imprime a precisão e o número de classificações incorretas.
7.  **Execução:**
    * Instancia o modelo.
    * Define o dispositivo de execução (GPU "cuda" se disponível, caso contrário CPU "cpu").
    * Move o modelo para o dispositivo selecionado.
    * Chama a função `treino`.
    * Chama a função `validacao` para avaliar o modelo treinado.

## Tecnologias Utilizadas

* Python 3
* PyTorch
* Torchvision
* NumPy
* Matplotlib

## Como Executar

1.  **Pré-requisitos:** Certifique-se de ter Python 3 e as bibliotecas listadas instaladas.
    ```bash
    pip install torch torchvision numpy matplotlib
    ```
2.  **Ambiente:** O código foi projetado para ser executado em um ambiente que suporte Jupyter Notebooks ou pode ser adaptado para um script Python (`.py`).
3.  **Execução:**
    * Se estiver usando um Jupyter Notebook (`.ipynb`), execute as células na ordem em que aparecem.
    * Se adaptado para um script Python, execute o script:
        ```bash
        python rede_neural.py
        ```
4.  **Dataset:** O dataset MNIST será baixado automaticamente na primeira execução para a pasta `./data`.
5.  **Dispositivo:** O código detectará automaticamente se uma GPU compatível com CUDA está disponível e a utilizará para acelerar o treinamento e a validação. Caso contrário, usará a CPU.

## Resultados (Exemplo)

Após 10 épocas de treinamento, o modelo atingiu a seguinte performance no conjunto de validação:

* **Precisão:** ~97.8%
* **Tempo de Treino:** ~63.8 segundos (este valor pode variar significativamente dependendo do hardware)