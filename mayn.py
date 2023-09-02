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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Criação da classe de partícula que irá definir as colisões elásticas e a 
# conservação do momento das nossas partículas de gases ideais.
class Particula:
    """Definição das características das partículas, especialmente sobre
    suas colisões elásticas dado o sistema sendo definido em função de 
    partículas de gases ideais.
    """

    def __init__(self, massa, raio, posicao, velocidade):
        """ Inicialização das características do nosso objeto  "Particula".

        ::mass::  a massa da partícula.
        ::raio::  o raio da partícula.
        ::posicao:: : o vetor de posição da partícula.
        ::velocidade::  o vetor de velocidade da partícula.
        """
        self.massa = massa
        self.raio = raio

        # Última posição e velocidade
        self.posicao = np.array(posicao)
        self.velocidade = np.array(velocidade)

        # Todas as posições e velocidades registradas durante a simulação.
        self.solpos = [np.copy(self.posicao)] # Inicializado com a posição incial da partícula.
        self.solvel = [np.copy(self.velocidade)] # Inicializado com a posição inicial da velocidade.
        self.solvel_mag = [np.linalg.norm(np.copy(self.velocidade))] # magnitude da velocidade inicial da partícula. 

    def calcular_passo(self, passo):
        """Calcula a posição do próximo passo."""
        self.posicao += passo * self.velocidade
        self.solpos.append(np.copy(self.posicao))
        self.solvel.append(np.copy(self.velocidade))
        self.solvel_mag.append(np.linalg.norm(np.copy(self.velocidade)))

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
        norma = np.linalg.norm(di)
        if norma - (r1 + r2) * 1.1 < 0:
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
        norma = np.linalg.norm(di)
        if norma - (r1 + r2) * 1.1 < passo * abs(np.dot(v1 - v2, di)) / norma:
            self.velocidade = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
            particula.velocidade = v2 - 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * (-di)

    def calcular_reflexo(self, passo, tamanho):
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
            self.velocidade[1] *= -1.


def resolver_passo(lista_particulas, passo, tamanho):
    """Resolve um passo para cada partícula."""

    # Detectar colisão e batida nas bordas de cada partícula
    for i in range(len(lista_particulas)):
        lista_particulas[i].calcular_reflexo(passo, tamanho)
        for j in range(i + 1, len(lista_particulas)):
            lista_particulas[i].calcular_colisao(lista_particulas[j], passo)

    # Calcular a posição de cada partícula
    for particula in lista_particulas:
        particula.calcular_passo(passo)


def inicializar_lista_aleatoria(N, raio, massa, tamanho):
    """Gere objetos Particula de N em uma maneira aleatória em uma lista.
    Esta função cria uma lista de partículas com características aleatórias.
    Cada partícula tem uma massa, raio, posição inicial e velocidade inicial
    aleatórias."""

    lista_particulas = []

    for i in range(N):

        v_mag = np.random.rand(1) * 6
        v_ang = np.random.rand(1) * 2 * np.pi
        v = np.append(v_mag * np.cos(v_ang), v_mag * np.sin(v_ang))

        colisao = True
        while colisao == True:

            colisao = False
            pos = raio + np.random.rand(2) * (tamanho - 2 * raio)
            nova_particula = Particula(massa, raio, pos, v)
            for j in range(len(lista_particulas)):

                colisao = nova_particula.verificar_colisao(lista_particulas[j])

                if colisao == True:
                    break

        lista_particulas.append(nova_particula)
    return lista_particulas


# CONDIÇÕES DA SIMULAÇÃO
# Dados Definidos pelas autoras
numero_de_particulas = 90
tamanho_caixa = 200.

# Você precisa de um tfin e stepnumber maiores para obter o estado de equilíbrio.
tfin = 10
numero_de_passos = 150

passo_tempo = tfin / numero_de_passos

lista_particulas = inicializar_lista_aleatoria(numero_de_particulas, raio= 4, massa=1.2e-23, tamanho=200)

# Calcular simulação (leva algum tempo se stepnumber e numero_de_particulas forem grandes)
for i in range(numero_de_passos):
    resolver_passo(lista_particulas, passo_tempo, tamanho_caixa)
    #print(i)


################################# PLOT DOS DADOS ###################################

# Vamos simular?? Para isso vamos de matplot!!
#Crie uma figura que determina o espaço do plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Simulação de um gás ideal", fontsize=16)  # Adicione um título geral para os subplots

# Figura 1 - Simulação Partículas
ax1.axis('equal')
ax1.axis([-1, 30, -1, 30])
ax1.xaxis.set_visible(True)
ax1.yaxis.set_visible(True)
ax1.set_xlim([0, tamanho_caixa])
ax1.set_ylim([0, tamanho_caixa])
ax1.set_xlabel("Eixo X")
ax1.set_ylabel("Eixo Y")
ax1.set_title(" Simulação bidimensional das partículas")  # Adicione um título para o gráfico 1


# Colocar na Figura 1 as partículas de acordo com a quantidade estabelecida anteriormente.
circulo = [None] * numero_de_particulas
for i in range(numero_de_particulas):
    circulo[i] = plt.Circle((lista_particulas[i].solpos[0][0], lista_particulas[i].solpos[0][1]),
                            lista_particulas[i].raio, ec="black", color='#9A32CD', lw=1.25, zorder=15)
    ax1.add_patch(circulo[i])

mod_vel = [lista_particulas[i].solvel_mag[0] for i in range(len(lista_particulas))]


# Calcular distribuição de Boltzmann 2D

# A energia total deve ser constante para qualquer índice de tempo
def energia_total(lista_particulas, indice):
    return sum([lista_particulas[i].massa / 2. * lista_particulas[i].solvel_mag[indice] ** 2 for i in range(len(lista_particulas))])

E = energia_total(lista_particulas, 0)
E_media = E / len(lista_particulas)
k = 1.38064852e-23
T = 2 * E_media / (2 * k)
m = lista_particulas[0].massa
v = np.linspace(0, 10, 120)


#Boltzman Experimental - Simulação
def atualizar(frame):
    if frame < numero_de_passos:
        resolver_passo(lista_particulas, passo_tempo, tamanho_caixa)
        for j in range(numero_de_particulas):
            circulo[j].center = lista_particulas[j].solpos[frame][0], lista_particulas[j].solpos[frame][1]
        mod_vel = [lista_particulas[j].solvel_mag[frame] for j in range(len(lista_particulas))]
        ax2.clear()
        ax2.hist(mod_vel, bins=30, ec='black', color='#9A32CD', density=True, label="Simulação Moana")
        ax2.set_xlabel("Velocidade (m/s)")
        ax2.set_ylabel("Frequência da Densidade")
        E = energia_total(lista_particulas, frame)
        E_media = E / len(lista_particulas)
        T = 2 * E_media / (2 * k)
        Bt = m * np.exp(-m * v ** 2 / (2 * T * k)) / (2 * np.pi * T * k) * 2 * np.pi * v
        ax2.plot(v, Bt, color='red', label="Distribuição de Maxwell–Boltzmann")
        ax2.legend(loc="upper right")
        ax2.set_title(" Distribuição de Maxwell-Boltzmann")  # Adicione um título para o gráfico 1



# Rodar a animação!!
ani = FuncAnimation(fig, atualizar, frames=numero_de_passos + 1, repeat=True, interval=50)
#ani.save('amauras_simulacao_gas.gif', writer='pillow') # Transforma a simulação em gif.
plt.show()