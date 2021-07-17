altHP
============

Calculadora gráfica construída sobre um terminal iPython com capacidades de cálculo com variáveis complexas e tratamentos de matrizes e vetores.

Este projeto iniciou devido ao meu desgosto em usar a calculadora HP50g. Com os recursos e notações do python, este script cria uma série de funções que simplificam o uso de funções do NumPy e SciPy a fim de substituir as funcionalidades das calculadoras gráficas.

O arquivo .bat executa o script em um terminal iPython e mantém o interpretador sendo executado. Caso alguma variável interna ou função seja sobrescrita, pode-se resetar a calculadora executando-se `quit()`.

Lista de funções:

```python
c_(re, im) : Gera um número complexo ou um array de complexos na forma retangular, onde "re" e "im" assumem valores float ou array/list de floats.

cp_(r, theta) : Análogo à função c_() mas gera os valores complexos na forma polar com raio "r" e ângulo "theta". De forma análoga tamb´m é possível passar floats, arrays/listas de floats nos parâmetros.

ret(z) : Retorna um tuple com os valores de "re" e "im" do/s valor/es complexo/s fornecido/s.

pol(z) : Retorna um tuple com os valores de "r" e "theta" do/s valor/es complexo/s fornecido/s.
    
abs(z) : Retorna apenas o módulo do/s valor/es complexo/s.
    
phase(z) : Retorna apenas o ângulo do/s valor/es complexo/s.
```

