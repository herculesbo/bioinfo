import numpy as np
import argparse
from datetime import datetime

# Config Pontuacao
MATCH    =  1; # +1 para match (ou BLOSUM62)
MISMATCH = -1; # -1 para mismatch (ou BLOSUM62)
GAP      = 0; # penalidade de criacao de gap
GAP2     = 0; # penalidade de extensao de gap

BLOSUM62={
    '*':{'*':1,'A':-4,'C':-4,'B':-4,'E':-4,'D':-4,'G':-4,'F':-4,'I':-4,'H':-4,'K':-4,'M':-4,'L':-4,'N':-4,'Q':-4,'P':-4,'S':-4,'R':-4,'T':-4,'W':-4,'V':-4,'Y':-4,'X':-4,'Z':-4},
    'A':{'*':-4,'A':4,'C':0,'B':-2,'E':-1,'D':-2,'G':0,'F':-2,'I':-1,'H':-2,'K':-1,'M':-1,'L':-1,'N':-2,'Q':-1,'P':-1,'S':1,'R':-1,'T':0,'W':-3,'V':0,'Y':-2,'X':-1,'Z':-1},
    'C':{'*':-4,'A':0,'C':9,'B':-3,'E':-4,'D':-3,'G':-3,'F':-2,'I':-1,'H':-3,'K':-3,'M':-1,'L':-1,'N':-3,'Q':-3,'P':-3,'S':-1,'R':-3,'T':-1,'W':-2,'V':-1,'Y':-2,'X':-1,'Z':-3},
    'B':{'*':-4,'A':-2,'C':-3,'B':4,'E':1,'D':4,'G':-1,'F':-3,'I':-3,'H':0,'K':0,'M':-3,'L':-4,'N':3,'Q':0,'P':-2,'S':0,'R':-1,'T':-1,'W':-4,'V':-3,'Y':-3,'X':-1,'Z':1},
    'E':{'*':-4,'A':-1,'C':-4,'B':1,'E':5,'D':2,'G':-2,'F':-3,'I':-3,'H':0,'K':1,'M':-2,'L':-3,'N':0,'Q':2,'P':-1,'S':0,'R':0,'T':-1,'W':-3,'V':-2,'Y':-2,'X':-1,'Z':4},
    'D':{'*':-4,'A':-2,'C':-3,'B':4,'E':2,'D':6,'G':-1,'F':-3,'I':-3,'H':-1,'K':-1,'M':-3,'L':-4,'N':1,'Q':0,'P':-1,'S':0,'R':-2,'T':-1,'W':-4,'V':-3,'Y':-3,'X':-1,'Z':1},
    'G':{'*':-4,'A':0,'C':-3,'B':-1,'E':-2,'D':-1,'G':6,'F':-3,'I':-4,'H':-2,'K':-2,'M':-3,'L':-4,'N':0,'Q':-2,'P':-2,'S':0,'R':-2,'T':-2,'W':-2,'V':-3,'Y':-3,'X':-1,'Z':-2},
    'F':{'*':-4,'A':-2,'C':-2,'B':-3,'E':-3,'D':-3,'G':-3,'F':6,'I':0,'H':-1,'K':-3,'M':0,'L':0,'N':-3,'Q':-3,'P':-4,'S':-2,'R':-3,'T':-2,'W':1,'V':-1,'Y':3,'X':-1,'Z':-3},
    'I':{'*':-4,'A':-1,'C':-1,'B':-3,'E':-3,'D':-3,'G':-4,'F':0,'I':4,'H':-3,'K':-3,'M':1,'L':2,'N':-3,'Q':-3,'P':-3,'S':-2,'R':-3,'T':-1,'W':-3,'V':3,'Y':-1,'X':-1,'Z':-3},
    'H':{'*':-4,'A':-2,'C':-3,'B':0,'E':0,'D':-1,'G':-2,'F':-1,'I':-3,'H':8,'K':-1,'M':-2,'L':-3,'N':1,'Q':0,'P':-2,'S':-1,'R':0,'T':-2,'W':-2,'V':-3,'Y':2,'X':-1,'Z':0},
    'K':{'*':-4,'A':-1,'C':-3,'B':0,'E':1,'D':-1,'G':-2,'F':-3,'I':-3,'H':-1,'K':5,'M':-1,'L':-2,'N':0,'Q':1,'P':-1,'S':0,'R':2,'T':-1,'W':-3,'V':-2,'Y':-2,'X':-1,'Z':1},
    'M':{'*':-4,'A':-1,'C':-1,'B':-3,'E':-2,'D':-3,'G':-3,'F':0,'I':1,'H':-2,'K':-1,'M':5,'L':2,'N':-2,'Q':0,'P':-2,'S':-1,'R':-1,'T':-1,'W':-1,'V':1,'Y':-1,'X':-1,'Z':-1},
    'L':{'*':-4,'A':-1,'C':-1,'B':-4,'E':-3,'D':-4,'G':-4,'F':0,'I':2,'H':-3,'K':-2,'M':2,'L':4,'N':-3,'Q':-2,'P':-3,'S':-2,'R':-2,'T':-1,'W':-2,'V':1,'Y':-1,'X':-1,'Z':-3},
    'N':{'*':-4,'A':-2,'C':-3,'B':3,'E':0,'D':1,'G':0,'F':-3,'I':-3,'H':1,'K':0,'M':-2,'L':-3,'N':6,'Q':0,'P':-2,'S':1,'R':0,'T':0,'W':-4,'V':-3,'Y':-2,'X':-1,'Z':0},
    'Q':{'*':-4,'A':-1,'C':-3,'B':0,'E':2,'D':0,'G':-2,'F':-3,'I':-3,'H':0,'K':1,'M':0,'L':-2,'N':0,'Q':5,'P':-1,'S':0,'R':1,'T':-1,'W':-2,'V':-2,'Y':-1,'X':-1,'Z':3},
    'P':{'*':-4,'A':-1,'C':-3,'B':-2,'E':-1,'D':-1,'G':-2,'F':-4,'I':-3,'H':-2,'K':-1,'M':-2,'L':-3,'N':-2,'Q':-1,'P':7,'S':-1,'R':-2,'T':-1,'W':-4,'V':-2,'Y':-3,'X':-1,'Z':-1},
    'S':{'*':-4,'A':1,'C':-1,'B':0,'E':0,'D':0,'G':0,'F':-2,'I':-2,'H':-1,'K':0,'M':-1,'L':-2,'N':1,'Q':0,'P':-1,'S':4,'R':-1,'T':1,'W':-3,'V':-2,'Y':-2,'X':-1,'Z':0},
    'R':{'*':-4,'A':-1,'C':-3,'B':-1,'E':0,'D':-2,'G':-2,'F':-3,'I':-3,'H':0,'K':2,'M':-1,'L':-2,'N':0,'Q':1,'P':-2,'S':-1,'R':5,'T':-1,'W':-3,'V':-3,'Y':-2,'X':-1,'Z':0},
    'T':{'*':-4,'A':0,'C':-1,'B':-1,'E':-1,'D':-1,'G':-2,'F':-2,'I':-1,'H':-2,'K':-1,'M':-1,'L':-1,'N':0,'Q':-1,'P':-1,'S':1,'R':-1,'T':5,'W':-2,'V':0,'Y':-2,'X':-1,'Z':-1},
    'W':{'*':-4,'A':-3,'C':-2,'B':-4,'E':-3,'D':-4,'G':-2,'F':1,'I':-3,'H':-2,'K':-3,'M':-1,'L':-2,'N':-4,'Q':-2,'P':-4,'S':-3,'R':-3,'T':-2,'W':11,'V':-3,'Y':2,'X':-1,'Z':-3},
    'V':{'*':-4,'A':0,'C':-1,'B':-3,'E':-2,'D':-3,'G':-3,'F':-1,'I':3,'H':-3,'K':-2,'M':1,'L':1,'N':-3,'Q':-2,'P':-2,'S':-2,'R':-3,'T':0,'W':-3,'V':4,'Y':-1,'X':-1,'Z':-2},
    'Y':{'*':-4,'A':-2,'C':-2,'B':-3,'E':-2,'D':-3,'G':-3,'F':3,'I':-1,'H':2,'K':-2,'M':-1,'L':-1,'N':-2,'Q':-1,'P':-3,'S':-2,'R':-2,'T':-2,'W':2,'V':-1,'Y':7,'X':-1,'Z':-2},
    'X':{'*':-4,'A':-1,'C':-1,'B':-1,'E':-1,'D':-1,'G':-1,'F':-1,'I':-1,'H':-1,'K':-1,'M':-1,'L':-1,'N':-1,'Q':-1,'P':-1,'S':-1,'R':-1,'T':-1,'W':-1,'V':-1,'Y':-1,'X':-1,'Z':-1},
    'Z':{'*':-4,'A':-1,'C':-3,'B':1,'E':4,'D':1,'G':-2,'F':-3,'I':-3,'H':0,'K':1,'M':-1,'L':-3,'N':0,'Q':3,'P':-1,'S':0,'R':0,'T':-1,'W':-3,'V':-2,'Y':-2,'X':-1,'Z':4}}


def NW_alinha(seq1,seq2):
    inicio = datetime.now()
    print(" -> Sequencias:")
    print(seq1)
    print(seq2)
    #Inicializacao das Matrizes
    matriz_pontuacao = np.zeros([len(seq2)+1,len(seq1)+1])
    matriz_trace = np.zeros([len(seq2)+1,len(seq1)+1],dtype=str)
    #Penalidades de Criacao ou Extensao
    for j in range(0,len(seq1)+1):
        matriz_pontuacao[0][j] = GAP2*j
        matriz_trace[0][j] = "L"
    for i in range(0,len(seq2)+1):
        matriz_pontuacao[i][0] = GAP2*i
        matriz_trace[i][0] = "C"
    matriz_pontuacao[0][0] = 0
    matriz_trace[0][0] = "N"
    print("Alinhando com Needleman-Wunsch...",end=" ")
    for i in range(1,len(seq2)+1):
        for j in range (1,len(seq1)+1):
            pontua_diagonal=0
            pontua_lado=0
            pontua_cima=0
            # Calculando Pontuacao de Match/Mismatch
            letra1 = seq1[j-1:j]
            letra2 = seq2[i-1:i]
            if (letra1 == letra2):
                pontua_diagonal = matriz_pontuacao[i-1][j-1] + BLOSUM62[letra1][letra2] #COM BLOSUM62
                #pontua_diagonal = matriz_pontuacao[i-1][j-1] + MATCH #SEM BLOSUM62
            else:
                pontua_diagonal = matriz_pontuacao[i-1][j-1] + BLOSUM62[letra1][letra2] #COM BLOSUM62
                #pontua_diagonal = matriz_pontuacao[i-1][j-1] + MISMATCH #SEM BLOSUM62
            # Calculando Pontuação dos GAPS
            # Aplica GAP Penalindade de Criação de GAPS se ocorreu MATCH anteriormente, caso contrario aplica Penalidade de Extensão de GAP
            if (matriz_trace[i-1][j] == "D"):
                pontua_cima   = matriz_pontuacao[i-1][j] + GAP
            else:
                pontua_cima   = matriz_pontuacao[i-1][j] + GAP2
            if (matriz_trace[i][j-1] == "D"):
                pontua_lado = matriz_pontuacao[i][j-1] + GAP
            else:
                pontua_lado = matriz_pontuacao[i][j-1] + GAP2
            # Escolhe a melhor pontuacao
            if (pontua_diagonal >= pontua_cima):
                if (pontua_diagonal >= pontua_lado):
                    matriz_pontuacao[i][j] = pontua_diagonal
                    matriz_trace[i][j] = "D"
                else:
                    matriz_pontuacao[i][j] = pontua_lado
                    matriz_trace[i][j] = "L"
            else:
                if (pontua_cima >= pontua_lado):
                    matriz_pontuacao[i][j] = pontua_cima
                    matriz_trace[i][j] = "C"
                else:
                    matriz_pontuacao[i][j] = pontua_lado
                    matriz_trace[i][j] = "L"
    #Inicializacao do alinhamento
    alinha1 = ""
    alinha2 = ""
    tracking = ""
    j = len(seq1)
    i = len(seq2)
    #percorrendo matriz
    pontuacao_tracking = []
    while matriz_trace[i][j] != "N":
        tracking = tracking + matriz_trace[i][j]
        pontuacao_tracking.append(matriz_pontuacao[i][j])
        if (matriz_trace[i][j] == "D"):
            alinha1 = alinha1 + seq1[j-1:j]
            alinha2 = alinha2 + seq2[i-1:i]
            i=i-1
            j=j-1
        elif (matriz_trace[i][j] == "L"):
            alinha1 = alinha1 + seq1[j-1:j]
            alinha2 = alinha2 + "-"
            j=j-1
        elif (matriz_trace[i][j] == "C"):
            alinha1 = alinha1 + "-"
            alinha2 = alinha2 + seq2[i-1:i]
            i=i-1
    print("Finalizado.")
    #print("Pontuação - Backtrace - Alinhamento:",end="\n")
    alinha1 = alinha1[::-1]
    alinha2 = alinha2[::-1]
    tracking = tracking[::-1]
    print(" -> Pontuação:")
    print(list(reversed(pontuacao_tracking)))
    print(" -> Backtrace:\n Legenda: D =  Diagonal, L = Lado, C = Cima")
    print(tracking)
    print(" -> Alinhamento:")
    print(alinha1)
    print(alinha2)
    print(" -> Pontuação:",matriz_pontuacao[len(seq2)][len(seq1)])
    print("Tempo de Execução:",datetime.now() - inicio)
    return (tracking, alinha1, alinha2)

def carrega_arquivo(e_arq):
    arq = open(e_arq, 'r') # Abre para leitura
    linhas = arq.readlines() # Lê as linhas e separa em um vetor
    lista_seq = [] # cria um novo array para pegar somente as linhas de interesse
    for linha in linhas:
        if linha.find('>') != 0: # ignora as linhas que começam com >
            lista_seq.append(linha)
    if len(lista_seq) == 2:
        seq1 = lista_seq[0].rstrip("\n")
        seq2 = lista_seq[1].rstrip("\n")
    else:
        print('ERRO. Limite de Sequencias no Arquivo = 2')
        exit()
    return seq1, seq2

def saida_arquivo(s_arq,ali1, ali2):
    f = open(s_arq, "w")
    f.write(ali1+"\n"+ali2)
    f.close()

def main():
    #Trabalho desenvolvido para a Disciplina de Bioinformática
    #Alunos: Gabriel Quintanilha Peixoto, Hércules Batista de Oliveira, Joicymara Santos Xavier, Kaíssa Pereira Barbosa, Wellen Quézia Bernardes Durães
    #Professora: Raquel Minardi
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--entrada", nargs="+", help="Entrada de arquivo FASTA com duas sequencias", required=True)
    parser.add_argument("-s", "--saida", nargs="+", help="Arquivo de Saida de Dados", required=True)

    args = parser.parse_args()

    arquivo = args.entrada[0]
    seq1, seq2 = carrega_arquivo(arquivo)
    saida = args.saida[0]
    track_feito, alinhado1, alinhado2 = NW_alinha(seq1, seq2)
    saida_arquivo(saida, alinhado1, alinhado2)

if __name__ == "__main__":
    main()
