NW_na_mao

Algoritmo Needleman-Wunsch para alinhamento de sequências,utilizando a matriz BLOSUM62.

Argumentos

-e ou –entrada = Entrada de arquivo FASTA com duas sequencias. Requerido para execução

-s ou –saida =Arquivo de Saída de Dados, Requerido para execução


Exemplo de Uso

python3 nw_na_mao.py -e corona.fasta -s saida.log


Configurações

O algoritmo permite a configuração em seu código fonte (linhas 5 a 9) da pontuação a ser
utilizada em match, mismatch e gaps. Por padrão o código é apresentado configurado para
utilizar a matriz BLOSUM62 (linha 65 a 69), ignorando os valores de MATCH e MISMATCH
informados.

# Config Pontuacao

MATCH = 1; # +1 para match (ou BLOSUM62)

MISMATCH = -1; # -1 para mismatch (ou BLOSUM62)

GAP = 0; # penalidade de criacao de gap

GAP2 = 0; # penalidade de extensao de gap

