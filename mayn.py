##############################################################################
#                                                                            #
#                                                                            #
#                                                                            #
#                  SIMULAÇÃO DE UM GÁS IDEAL - CINÉTICA QUÍMICA              #
#                            ANA BRANDÃO & MONYQUE SILVA                     #
#                               PROFº AMAURI DE PAULA                        #
#                         inspirado em @rafael.fuente                        #                
#                                                                            #
#                                                                            #
##############################################################################

#Importação das bibliotecas
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
import matplotlib.animation as FuncAnimation
from scipy.optimize import curve_fit
import pandas as pd

class Particula:
    """Definição das características das partículas, especialmente sobre
    suas colisões elásticas dado o sistema sendo definido em função de 
    partículas de gases ideais.
    """
    
    def __init__(self, posicao, velocidade, raio, massa, reatividade):
        """ Inicialização das características do nosso objeto  "Particula".

        ::mass::  a massa da partícula.
        ::raio::  o raio da partícula.
        ::posicao:: : o vetor de posição da partícula.
        ::velocidade::  o vetor de velocidade da partícula.
        ::reatividade::  reatividade da partícula.

        """
        self.existe = True
        self.massa = massa
        self.raio = raio
        self.reatividade = reatividade

        # Última posição e velocidade das partículas
        self.posicao = np.array(posicao)
        self.velocidade = np.array(velocidade)

        # Características iniciais das partículas
        self.tipo = '#96C3EB'
        self.lista_tipo = ['#96C3EB']
        self.sopos = [np.copy(self.posicao)]
        self.solvel = [np.copy(self.velocidade)]
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocidade))]
        self.num_col = 0
        self.lista_existe = [1]

    def calcular_passo(self, passo):
        """Calcula a posição do próximo passo."""

        self.posicao += passo * self.velocidade
        self.sopos.append(np.copy(self.posicao))
        self.solvel.append(np.copy(self.velocidade))
        self.solvel_mag.append(np.linalg.norm(np.copy(self.velocidade)))
        self.lista_tipo.append(self.tipo)
        if self.existe:
            self.lista_existe.append(1)
        else:
            self.lista_existe.append(0)

    def verificar_colisao(self, particula):
        """Verifica se há colisão com outra partícula.
        A colisão é considerada quando as distâncias entre os
        centros das duas partículas (levando em conta seus raios)
        são menores do que a soma dos raios multiplicada por 1.1. 
        Se essa condição for atendida, o método retorna True, 
        indicando que ocorreu uma colisão. Caso contrário, ele 
        retorna False, indicando que não houve colisão."""

        r1, r2 = self.raio, particula.raio
        x1, x2 = self.posicao, particula.posicao
        di = x2 - x1
        norm = np.linalg.norm(di)
        if self.existe and particula.existe and norm - (r1 + r2) * 1.1 < 0:
            return True
        else:
            return False

    def calcular_colisao(self, particula, passo):
        """Calcula a velocidade após a colisão com outra partícula."""

        m1, m2 = self.massa, particula.massa
        r1, r2 = self.raio, particula.raio
        v1, v2 = self.velocidade, particula.velocidade
        x1, x2 = self.posicao, particula.posicao
        di = x2 - x1
        norm = np.linalg.norm(di)
        react = rd.random()
        if norm - (r1 + r2) * 1.1 < passo * abs(np.dot(v1 - v2, di)) / norm:
            if (self.reatividade + particula.reatividade) / 2 < react or self.tipo == '#9A32CD' or particula.tipo == '#9A32CD':
                self.num_col += 1
                particula.num_col += 1
                self.velocidade = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
                particula.velocidade = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * (-di)
            if (self.reatividade + particula.reatividade) / 2 >= react and self.tipo == '#96C3EB' and particula.tipo == '#96C3EB':
                self.tipo = '#9A32CD'
                self.massa = m1 + m2
                self.raio = np.sqrt(r1 ** 2 + r2 ** 2)
                self.velocidade = np.array([(m1 * v1[0] + m2 * v2[0]) / (m1 + m2), (m1 * v1[1] + m2 * v2[1]) / (m1 + m2)])
                particula.existe = False

    def calcular_colisao_parede(self, passo, tamanho):
        """Calcula a velocidade após bater em uma borda.
        ::passo::  o passo de cálculo.
        ::tamanho::  o tamanho do meio.
        """
        r, v, x = self.raio, self.velocidade, self.posicao
        projx = passo * abs(np.dot(v, np.array([1., 0.])))
        projy = passo * abs(np.dot(v, np.array([0., 1.])))
        if abs(x[0]) - r < projx or abs(tamanho - x[0]) - r < projx:
            self.velocidade[0] *= -1
        if abs(x[1]) - r < projy or abs(tamanho - x[1]) - r < projy:
            self.velocidade[1] *= -1

def calcular_passo(particulas, passo, tamanho):
    """Resolve um passo para cada partícula."""

    for i in range(len(particulas)):
        particulas[i].calcular_colisao_parede(passo, tamanho)
        for j in range(i + 1, len(particulas)):
            particulas[i].calcular_colisao(particulas[j], passo)
    for particula in particulas:
        particula.calcular_passo(passo)

def inicializar_lista_aleatoria(N, raio, massa, tamanho_caixa, reatividade):
    """Gere objetos Particula de N em uma maneira aleatória em uma lista.
    Esta função cria uma lista de partículas com características aleatórias.
    Cada partícula tem uma massa, raio, posição inicial e velocidade inicial
    aleatórias."""
    particulas = []
    for i in range(N):
        v_mag = np.random.rand(1) * 20
        v_ang = np.random.rand(1) * 2 * np.pi
        v = np.append(v_mag * np.cos(v_ang), v_mag * np.sin(v_ang))
        colisao = True
        while colisao:
            colisao = False
            pos = raio + np.random.rand(2) * (tamanho_caixa - 2 * raio)
            nova_particula = Particula(pos, v, raio, massa, reatividade)
            for j in range(len(particulas)):
                colisao = nova_particula.verificar_colisao(particulas[j])
                if colisao:
                    break
        particulas.append(nova_particula)
    return particulas

# Condições de Contorno
numero_particulas = 100
tamanho_caixa = 200
massa = 1.2e-23
raio = 1
reatividade = 1

# Parâmetros simulação
TFIM = 10
passinhos = 1000
STEP = TFIM / passinhos
PARTICULAS = inicializar_lista_aleatoria(numero_particulas, raio=raio, massa=massa, tamanho_caixa=tamanho_caixa, reatividade=reatividade)

print(len(PARTICULAS))
for i in range(passinhos):
    calcular_passo(PARTICULAS, STEP, tamanho_caixa)

trajetorias = []
existencia = []
cores = []

for i in PARTICULAS:
    traj = list(i.sopos)
    exist = list(i.lista_existe)
    lista_cores = list(i.lista_tipo)
    trajetorias.append(traj)
    existencia.append(exist)
    cores.append(lista_cores)
ims = []


##############################################################
################# PLOT DOS DADOS - SIMULAÇÃO #################

# Inicialização da Figura
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Simulação de um gás ideal", fontsize=16)

# Definição dos eixos da Figura 1
ax1.axis('equal')
ax1.axis([-1, 30, -1, 30])
ax1.xaxis.set_visible(True)
ax1.yaxis.set_visible(True)
ax1.set_xlim([0, tamanho_caixa])
ax1.set_ylim([0, tamanho_caixa])
ax1.set_xlabel("Eixo X")
ax1.set_ylabel("Eixo Y")
ax1.set_title("Simulação bidimensional das partículas")

# Definição dos eixos da Figura 2
ax2.set_xlabel("Tempo[s]")
ax2.set_ylabel("Concentração")
ax2.set_title("Concentração de partículas ao longo do tempo")


# Criação do Maxwell-Boztman
def energia_total(lista_particulas, indice):
    return sum([lista_particulas[i].massa / 2. * lista_particulas[i].solvel_mag[indice] ** 2 for i in range(len(lista_particulas))])

E = energia_total(PARTICULAS, 0)
E_media = E / len(PARTICULAS)
k = 1.38064852e-23
T = 2 * E_media / (2 * k)
m = PARTICULAS[0].massa
v = np.linspace(0, 28, 240)


#Lista para contagem das partículas
num_particulas_A = []
num_particulas_B = []

def atualizar_animacao(frame):
    "Função de animação da simulação"

    # Chamar listas globais
    global PARTICULAS, ims, num_particulas_A, num_particulas_B
    
    # Calcular o próximo passo da colisão
    calcular_passo(PARTICULAS, STEP, tamanho_caixa)

    # Preparo da Figura 1
    ax1.clear()
    ax1.axis('equal')
    ax1.axis([-1, 30, -1, 30])
    ax1.xaxis.set_visible(True)
    ax1.yaxis.set_visible(True)
    ax1.set_xlim([0, tamanho_caixa])
    ax1.set_ylim([0, tamanho_caixa])
    ax1.set_xlabel("Eixo X")
    ax1.set_ylabel("Eixo Y")
    ax1.set_title("Simulação bidimensional das partículas")
    
    #Inicializar listas do círculo e dos contadores
    circles = []
    contador_A = 0
    contador_B = 0

    # Plot das partículas
    for h in range(len(trajetorias)):
        x = trajetorias[h][frame][0]
        y = trajetorias[h][frame][1]
        e = existencia[h][frame]
        cor = cores[h][frame]
        if e == 1:
            circle = plt.Circle((x, y), raio, fill=True, color=cor)
            ax1.add_artist(circle)
            circles.append(circle)
            if cor == '#96C3EB':
                contador_A = contador_A + 1
                num_particulas_A.append(contador_A)
            elif cor == '#9A32CD':
                contador_B = contador_B + 1
                num_particulas_B.append(contador_B)
    ims.append(circles)
    
    # Atualização do contador em tempo real
    ax1.text(0.67, 0.90, f"Partículas A: {contador_A}", transform=ax1.transAxes, fontsize=12, color='#96C3EB')
    ax1.text(0.67, 0.82, f"Partículas B: {contador_B}", transform=ax1.transAxes, fontsize=12, color='#9A32CD')

    # Atualização da simulação da Figura 2
    ax2.clear()
    ax2.set_xlabel("Velocidade (m/s)")
    ax2.set_ylabel("Frequência da Densidade")
    ax2.set_title("Distribuição de Maxwell-Boltzmann")
    
    #Definição do Maxwell-Boltzmann da Figura 2
    velocidades_timestep = [np.linalg.norm(particula.velocidade) for particula in PARTICULAS]
    ax2.hist(velocidades_timestep, bins=30, density=True, color='b', alpha=0.5)
    Bt = m * np.exp(-m * v ** 2 / (2 * T * k)) / (2 * np.pi * T * k) * 2 * np.pi * v
    ax2.plot(v, Bt, color='red', label="Distribuição de Maxwell–Boltzmann")
    ax2.legend()



# Roda a simulação
ani = animation.FuncAnimation(fig, atualizar_animacao, frames=passinhos + 1, interval=40, repeat=True)


# Criação da Figura 3 - sem animação!

for k in range(passinhos):
    a = 0
    b = 0
    for h in range(len(trajetorias)):
        cor = cores[h][k]
        e = existencia[h][k]
        if cor == '#96C3EB' and e == 1:
            a = a + 1
        if cor == '#9A32CD' and e == 1:
            b = b + 1
    num_particulas_A.append(a)
    num_particulas_B.append(b)


# Obtenção da quantidade de partículas em frames e conversão desta para um dataframe
df = pd.DataFrame({"Passos": range(passinhos), "Partículas A": num_particulas_A, 'Partículas B': num_particulas_B})
print(df)

# Exponencial
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Logarítmo
def funcaozinha(x, a, b, c):
    return a * np.log(b * x) + c

# Chama as listas
y_data = df['Partículas A']
x_data = df['Passos']
y2_data = df['Partículas B']

popt, pcov = curve_fit(func, x_data, y_data)
popti, pcov = curve_fit(funcaozinha, x_data, y2_data)

# Cria a Figura 3 e a Figura 4
fig, ((ax3, ax4)) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Simulação da lei de velocidade de reação", fontsize=16)
plt.tight_layout()

# Cria os eixos da Figura
ax3.plot(x_data, func(x_data, *popt), 'r--', label="Curva Ajustada A")
ax3.plot(x_data, func(x_data, *popti), 'r--', label="Curva Ajustada B")
ax3.grid('- -')
ax3.plot(range(passinhos), num_particulas_A, linewidth=2.5, label='A', color='#96C3EB')
ax3.plot(range(passinhos), num_particulas_B, linewidth=2.5, label='B', color='#9A32CD')
ax3.legend()

plt.show()
