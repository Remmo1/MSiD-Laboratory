# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.8.10 64-bit
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dopasowanie rozkładu do danych
# ### Tymoteusz Trętowicz, 260451

# %%
from collections import Counter, OrderedDict
from io import BytesIO
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# %% [markdown]
# ## Dane
# Dana są pobierane ze strony https://archive.ics.uci.edu/ml/, która gromadzi dane w celach usdostępniania ich do projektów tyczących się uczenia maszynowego.
# Dane, które zostały wybrane to informacje odnośnie ludzi, które zostały zdobyte przez portugalski bank w trakcie kampani telefonicznej w okresie od maja 2008 do czerwca 2013. Dane te były gromadzone na potrzebę badania: https://core.ac.uk/download/pdf/55631291.pdf

# %%
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
resp = requests.get(URL).content

zipfile = ZipFile(BytesIO(resp))
zipfile.extract("bank.csv")
df = pd.read_csv('bank.csv', sep=';')
df


# %% [markdown]
# Analizować będziemy ilość osób w danym wieku.
# $K$ to wektor par ( wiek : ilość osób w tym wieku ).
# Odpowiednio $X$ to wektor, zawierający informację o ilości osób w danym wieku, a $Y$ to wektor zawierający wiek tych osób.

# %%
Q = OrderedDict(sorted(Counter(df['age'].values).items()))
X = list(Q.values())
Y = list(Q.keys())
Q


# %% [markdown]
# Pierwszy standardowy test sprawdzający normalność danych to test Shapiro-Wilka: na podstawie statystyki $W$, danej wzorem:
# $$W = \frac{\left( \sum^{n}_{i=1} a_i x_{(i)} \right)^2}{\sum^{n}_{i=1} \left( x_i - \overline{x} \right)^2}$$ 
# gdzie: $x_{(i)}$ to i-ta najmniejsza wartość, $\overline{x} = \frac{\left( x_1 + \cdots + x_n \right)}{n}$ jest wartością średnią, oraz $a$ są stabelaryzowanymi wartościami, można sprawdzić hipotezę zerową oraz alternatywną:
#
# $H_0$: Próba pochodzi z populacji o rozkładzie normalnym
#
# $H_1$: Próba nie pochodzi z populacji o rozkładzie normalnym.
#
# #### Interpretacja
# $\textit{p-wartość}$ jest prawdopodobieństwem, że wynik testu jest przytajmniej tak samo ekstremalny jak wynik zaobserwowany, zakładając, że hipoteza zerowa jest poprawna.
#
# Jeżeli $\textit{p-wartość}$ jest mniejsza niż wybrana wartość krytyczna hipoteza zerowa jest odrzucona, co znaczy, że dane nie pochodzą ze zbioru o rozkładzie normalnym. Natomiast jeżeli $\textit{p-wartość}$ jest większa, niż wybrana wartość krytyczna, hipoteza zerowa nie może być odrzucona.

# %% [markdown]
# ### Test Shapiro-Wilka dla danego zbiorów

# %%
from scipy import stats

x_res_shap = stats.shapiro(X)

print("Wyniki testu Shapiro-Wilka: ")
print('statystyka: %s, p-wartość: %s' % ("{:.8f}".format(x_res_shap[0]), "{:.45f}".format(x_res_shap[1])))

# %% [markdown]
# ### Test d'Agostino-Pearsona dla podanych zbiorów

# %%
x_res_agost = stats.normaltest(X)

print("Wyniki testu d'Agostino-Pearsona: ")
print('statystyka: %s, p-wartość: %s' % ("{:.8f}".format(x_res_agost[0]), "{:.8f}".format(x_res_agost[1])))


# %% [markdown]
# W obu testach p-wartość jest poniżej wartości krytycznej $= 0.05$, więc na tej podstawie odrzucam hipotezę zerową. Stąd stwierdzam, że dane nie pochodzą ze zbioru o rozkładzie normalnym.

# %% [markdown]
# ### Estymator Parzena
#
# Funkcja jądrowa jest dana wzorem: $$K(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$
# Stąd estymator Parzena:
# $$\hat{f}(x) = \frac{1}{Nh^d} \sum^{N}_{i = 1} K\left( \frac{x - x_i}{h} \right)$$
#
# gdzie $d$ oznacza wymiarowość wektora $X$, $N$ rozmiar próby, a $h$ to szerokość okna. Tutaj $d = 1$

# %%
def K(x):
  return np.exp(-x**2/2)/np.sqrt(2*np.pi)

def parzen_est(h, values, x_axis):
  sum = 0
  for mes in values:
    arg = (x_axis - mes)/h
    sum += K(arg)
  
  return sum / (h * len(values))


# %% [markdown]
# Przed użyciem estymatora Parzena wykonujemy normalizację danych, tak, żeby wartości zbioru były z przedziału $D = \left[ 0; 1\right]$.
#
# Osiągamy to dzieląc każdy element zbioru przez maksymalną wartość zbioru.
# $$XN = \frac{X}{x_{(N)}}$$

# %%
XN = list(map(lambda x: x/max(X), X))
XN

# %% [markdown]
# W ten sposób estymujemy funkcję gęstości prawdopodobieństwa dla trzech szerokości okna $h$.

# %%
plt.figure(figsize=(10,4))

H = [0.4, 0.2, 0.1]

os_x = np.linspace(min(XN)-1, max(XN)+1, num=300)
for h in H:
  os_y = parzen_est(h, XN, os_x)
  plt.plot(os_x, os_y, label=f'h = {h}')

plt.plot(XN, np.zeros_like(XN), 's', markersize=1, color='black', label='dane unormalizowane')
plt.xlabel('$x$ (unormalizowane)', fontsize=14)
plt.ylabel('$f$', fontsize=14, rotation='horizontal', labelpad=15)
plt.legend(fontsize=12, shadow=True)
plt.show()
