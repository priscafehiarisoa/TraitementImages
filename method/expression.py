import re
from inspect import signature

import numpy as np
from matplotlib import pyplot as plt


def is_number(char):
    try:
        if str(char).endswith('x'):
            char = char.removesuffix('x')
            if char == '-' or char == '':
                char += '1'
        float(char)
        return True
    except ValueError:
        return False


def priority(operator):
    operators = {
        '*': 2,
        '/': 2,
        '//': 2,
        '/-': 2,
        '-': 1,
        '+': 1,
    }
    try:
        return operators[operator]
    except KeyError:
        try:
            test = globals()[operator]
            return 3
        except Exception:
            raise SyntaxError(f"L'operateur ou la fonction '{operator}' n'existe pas!")


def clean(expression):
    expression = verify(expression)
    clean_exp = []
    temp = ""
    previous_type = ""
    for x in expression:
        if x.isspace():
            continue

        if x == 'x':
            current_type = "x"
        elif x.isalpha():
            current_type = "alpha"
        elif is_number(x) or x == '.':
            current_type = "number"
        elif x == '(' or x == ')':
            current_type = "pr"
        else:
            current_type = "operator"

        if current_type == 'pr' and previous_type == 'pr':
            clean_exp.append(x)
            temp = ""

        if current_type == 'x':
            temp += x
        elif current_type == previous_type or previous_type == '':
            temp += x
        else:
            if (temp == '-' or temp == '+') \
                    and ((len(clean_exp) > 0 and clean_exp[-1] != ')' and not is_number(clean_exp[-1]))
                         or len(clean_exp) == 0):
                clean_exp.append('0')
            if len(temp) > 0:
                clean_exp.append(temp)
            temp = x
        previous_type = current_type
    if len(temp) > 0:
        clean_exp.append(str(temp))

    return clean_exp


def verify(expression):
    count_p = 0
    i_last_p = 0

    for i, x in enumerate(expression):
        if x == '(':
            i_last_p = i
            count_p += 1
        elif x == ')':
            count_p -= 1

        if count_p < 0:
            error = f"Cette parenthèse n'a pas été ouverte!\n{expression}\n" + ("-" * i) + '^'
            raise SyntaxError(error)

    if count_p > 0:
        error = f"Il y {count_p} parenthèse(s) non-fermée(s)!\n{expression}\n" + ("-" * i_last_p) + '^'
        raise SyntaxError(error)
    return expression


def convert(expression):
    expression = clean(expression)
    pile = []
    output = []
    for elem in expression:
        if is_number(elem):
            output.append(elem)
        elif elem == ',':
            while pile[0] != '(':
                output.append(pile.pop(0))
        elif elem == '(':
            pile.insert(0, elem)
        elif elem == ')':
            while pile[0] != '(':
                output.append(pile.pop(0))
            pile.pop(0)
        else:
            while len(pile) > 0 and pile[0] != '(' and priority(elem) <= priority(pile[0]):
                output.append(pile[0])
                pile.pop(0)
            pile.insert(0, elem)
    if len(pile) > 0:
        output.extend(pile)
    return output


def pre_operation(a, b):
    nbs = []
    if not is_number(str(a)):
        nbs.extend(a)
    else:
        nbs.append(a)
    if not is_number(str(b)):
        nbs.extend(b)
    else:
        nbs.append(b)
    return nbs


def plus(a, b):
    nbs = pre_operation(a, b)
    exp_x = []
    exp_c = []
    for nb in nbs:
        if str(nb).endswith('x'):
            t = str(nb).removesuffix('x')
            if t == '' or t == '-':
                t += '1'
            exp_x.append(float(t))
        else:
            exp_c.append(float(nb))
    s_exp_x = sum(exp_x)
    s_exp_c = sum(exp_c)
    if s_exp_x != 0 and s_exp_c != 0:
        return [str(s_exp_x) + 'x', s_exp_c]
    if s_exp_x == 0:
        return str(s_exp_c)
    return str(s_exp_x) + 'x'


def minus(a, b):
    if not is_number(str(b)):
        new_b = []
        for elem in b:
            if str(elem).startswith('-'):
                new_b.append(str(elem).removeprefix('-'))
            else:
                new_b.append('-' + str(elem))
    else:
        new_b = '-' + str(b)
    return plus(a, new_b)


def multiply(a, b):
    a = [a] if is_number(str(a)) else a
    b = [b] if is_number(str(b)) else b
    res = []
    for elem_a in a:
        for elem_b in b:
            if str(elem_a).endswith('x') and str(elem_b).endswith('x'):
                if str(elem_a).removesuffix('x') == '0' or str(elem_b).removesuffix('x') == '0':
                    res.append('0')
                else:
                    raise SyntaxError(f"Ne supporte pas la multiplication de deux variables! {elem_a} * {elem_b}")
            elif str(elem_a).endswith('x') and not str(elem_b).endswith('x'):
                if str(elem_a).removesuffix('x') == '':
                    res.append(str(elem_b) + 'x')
                else:
                    res.append(str(float(str(elem_a).removesuffix('x')) * float(str(elem_b))) + 'x')
            elif not str(elem_a).endswith('x') and str(elem_b).endswith('x'):
                if str(elem_b).removesuffix('x') == '':
                    res.append(str(elem_a) + 'x')
                else:
                    res.append(str(float(str(elem_b).removesuffix('x')) * float(str(elem_a))) + 'x')
            else:
                res.append(float(elem_a) * float(elem_b))
    return res


def divide(a, b):
    a = [a] if is_number(str(a)) else a
    b = [b] if is_number(str(b)) else b
    res = []
    for elem_a in a:
        for elem_b in b:
            if str(elem_a).endswith('x') and str(elem_b).endswith('x'):
                if str(elem_b).removesuffix('x') == '0':
                    raise ZeroDivisionError(f"Impossible de diviser par 0! {elem_a} / {elem_b}")
                res.append(str(float(str(elem_a).removesuffix('x')) / float(str(elem_b).removesuffix('x'))))
            elif str(elem_a).endswith('x') and not str(elem_b).endswith('x'):
                if str(elem_a).removesuffix('x') == '':
                    res.append(str(elem_b) + 'x')
                else:
                    res.append(str(float(str(elem_a).removesuffix('x')) / float(str(elem_b))) + 'x')
            elif not str(elem_a).endswith('x') and str(elem_b).endswith('x'):
                if str(elem_b).removesuffix('x') == '':
                    res.append(str(elem_a) + 'x')
                else:
                    res.append(str(float(str(elem_a)) / float(str(elem_b).removesuffix('x'))) + 'x')
            else:
                res.append(float(elem_a) / float(elem_b))
    return res


# def calculate(expression):
#     try:
#         expression = convert(expression)
#     except Exception:
#         raise SyntaxError('Il y a une erreur dans votre expression!')
#     pile = []
#     for elem in expression:
#         if is_number(elem):
#             pile.insert(0, elem)
#         else:
#             temp = 0
#             operations = {'+': lambda x, y: plus(x, y),
#                           '-': lambda x, y: minus(x, y),
#                           '*': lambda x, y: multiply(x, y),
#                           '/': lambda x, y: divide(x, y),
#                           '/-': lambda x, y: divide(x, '-' + str(y)),
#                           }
#             if elem in operations:
#                 op = operations[elem]
#                 temp = op(pile[1], pile[0])
#                 pile.pop(0)
#                 pile.pop(0)
#             elif elem.isalpha():
#                 function = globals()[elem]
#                 try:
#                     sign = signature(function)
#                     r_pile = pile[::-1]
#                     temp = function(*[float(x) for x in r_pile[len(r_pile) - len(sign.parameters):]])
#                     for i in range(len(sign.parameters)):
#                         pile.pop(0)
#                 except Exception:
#                     if is_number(function):
#                         temp = function
#                     else:
#                         raise SyntaxError(f"La fonction '{elem}' n'existe pas!")
#             pile.insert(0, temp)
#     if isinstance(pile[0], list):
#         print(pile[0])
#         return result(pile[0])
#     else:
#         print(pile)
#         return result(pile)

def calculate(expression):
    try:
        expression = convert(expression)
    except Exception:
        raise SyntaxError('Il y a une erreur dans votre expression!')
    pile = []
    for elem in expression:
        if is_number(elem):
            pile.insert(0, elem)
        else:
            temp = 0
            operations = {'+': lambda x, y: plus(x, y),
                          '-': lambda x, y: minus(x, y),
                          '*': lambda x, y: multiply(x, y),
                          '/': lambda x, y: divide(x, y),
                          '/-': lambda x, y: divide(x, '-' + str(y)),
                          }
            if elem in operations:
                op = operations[elem]
                temp = op(pile[1], pile[0])
                pile.pop(0)
                pile.pop(0)
            elif elem.isalpha():
                function = globals()[elem]
                try:
                    sign = signature(function)
                    r_pile = pile[::-1]
                    temp = function(*[float(x) for x in r_pile[len(r_pile) - len(sign.parameters):]])
                    for i in range(len(sign.parameters)):
                        pile.pop(0)
                except Exception:
                    if is_number(function):
                        temp = function
                    else:
                        raise SyntaxError(f"La fonction '{elem}' n'existe pas!")
            pile.insert(0, temp)
    if(isinstance(pile,list)):

        if isinstance(pile[0], list):
            print(pile[0])
            return pile[0]
        else:
            print(pile)
            return pile



def result(res):
    if is_number(str(res)):
        return res
    r = []
    for elem in res:
        if str(elem).startswith('-'):
            r.append(str(elem))
        else:
            r.append('+ ' + str(elem))
    return ' '.join(r).strip()


def est_nombre(nombre):
    try:
        float(nombre)
        return True
    except Exception:
        return False


def est_operateur(valeur):
    op = ['+', '-', '*', '/']
    return valeur in op


def formater(equation):
    res = ""
    for i, x in enumerate(equation):
        if x == " ":
            if (est_nombre(res[-1]) and (est_nombre(equation[i + 1]) or equation[i + 1] == "x")) \
                    or est_operateur(res[-1]):
                continue
        res += x
    return res


def inverser(valeur):
    if str(valeur).startswith('+'):
        return '-' + valeur[1:]
    if str(valeur).startswith('-'):
        return '+' + valeur[1:]
    if str(valeur).startswith('*'):
        return '/' + valeur[1:]
    if str(valeur).startswith('/'):
        return '*' + valeur[1:]


def regler_signe(list_valeur):
    res = []
    for x in list_valeur:
        if not est_operateur(x[:1]):
            res.append("+" + str(x))
        else:
            res.append(x)
    return res


def separation(equation):
    equation = formater(equation)
    x = []
    c = []
    apres_egal = False
    for i, elem in enumerate(str(equation).split(" ")):
        if elem == '=':
            apres_egal = True
            continue
        if str(elem).endswith('x'):
            if apres_egal:
                x.append(inverser(elem))
            else:
                x.append(elem)
        else:
            if apres_egal:
                c.append(elem)
            else:
                c.append(inverser(elem))
    return f"{' '.join(regler_signe(x))} = {' '.join(regler_signe(c))}"
# ----------------------------formater l'equation pour pouvoir caire le calcul

def distribution_de_l_equation(equation):
    operator = ["+","-","=","*","/"]
    equation = formater(equation,operator)
    partie1 = equation.split("=")[0]
    partie2 = equation.split("=")[1]
    partie2 = "="+partie2
    if "x" in partie1:
        equation = transposer(partie1,partie2)
    return equation

def findOperator(part,operator):
    for i in range(len(part)):
        if part[i] == operator:
            return i
    return False

def formater(equation,operator):
    for i in range(0,len(equation),2):
        if(i+1<len(equation)):
            if(equation[i+1] not in operator):
                partie1 = equation[:i+1]
                partie2 = equation[i+1:]
                equation = partie1+partie2
                print(equation)
    return equation

def findInverse(operator):
    operand =["+","-","-","+","*","/","/","*"]
    for i in range(0,len(operand),2):
        if operand[i] == operator:
            return operand[i+1]


def transposer(partie1, partie2):
    if not findOperator(partie1, "+") == False:
        operator = findOperator(partie1, "+")
        operand = partie1[operator]
        chiffre = partie1[operator + 1]
        inv = findInverse(operand)
        partie2 = partie2 + inv + chiffre
        partie1 = partie1[:operator] + partie1[operator + 2:]

    if not findOperator(partie1, "-") == False:
        operator = findOperator(partie1, "-")
        operand = partie1[operator]
        chiffre = partie1[operator + 1]
        inv = findInverse(operand)
        partie2 = partie2 + inv + chiffre
        partie1 = partie1[:operator] + partie1[operator + 2:]
    partie2 = partie2[1:]
    partie2 = "(" + partie2 + ")"

    if not findOperator(partie1, "*") == False:
        operator = findOperator(partie1, "*")
        operand = partie1[operator]
        chiffre = partie1[operator - 1]
        inv = findInverse(operand)
        partie2 = partie2 + inv + chiffre
        partie2 = "(" + partie2 + ")"
        partie1 = partie1[operator + 1:]

    if not findOperator(partie1, "/") == False:
        operator = findOperator(partie1, "/")
        operand = partie1[operator]
        chiffre = partie1[operator + 1]
        inv = findInverse(operand)
        partie2 = partie2 + inv + chiffre
        partie1 = partie1[:operator] + partie1[operator + 2:]
    partie2 = "=" + partie2

    return partie1 + partie2



def isOperator(operators):
    if(operators in "+=-/"):
        return True
    return False


# ----------resolution
def getSigneOp(equation):
    equation=str(equation)
    if(equation.__contains__("<")):
        return "<"
    elif equation.__contains__(">"):
        return ">"

    elif equation.__contains__("="):
        return "="
def arranger_l_equation(equation):
    # etape 1 transformer a en x
    equation=str(equation).replace('A','x')

    equation2=""
#     etape 2 transformer < et > en =
    equation=equation.replace('<','=')
    equation=equation.replace('>','=')
    print(equation)

    # diviser l'equation en deux s'il y a un truc qu'on a oublié
    wer=equation.split("=")
    # inversion de la position de x s'il y a un qu'on a oublié dans la partie 2
    if(wer[1].__contains__('x')):
        print("true")
        res=wer[1].split('x')
        i=len(res[0])-1
        response=""
        print(f"ddssdsdsds {i}")
        print(f"ddssdsdsds {len(res[0])}")
        print(f"ddssdsdsds {(res[0])}")
        operat=""
        while(isOperator(res[0][i])==False):
            response+=res[0][i]
            i-=1
            if(i<=0 and isOperator(res[0][i])==False):
                break
        if(isOperator(res[0][i])==False)    :
            operat = "+"
        else:
            operat=res[0][i]

        # effacer la partie ou il y a le x dans la deuxieme partie de la fonction
        if operat=='=':
            operat="+"
            wer[1]=wer[1].replace(response[::-1]+"x","")
        else:
            resp=response+ operat
            wer[1]=wer[1].replace(resp[::-1]+"x", "")
            print(f"resp ilany{wer[1]}")

        #     inverser le signe de la reponse
        operat=findInverse(operat)
        response+=operat
        response=response[::-1]

        # ajouter l'inverse dans la partie opposé
        wer[0]+=response+"x"

        # on obtient
        equation2=wer[0]+"="+wer[1]
        print(f"dd {equation2}")
    return equation2

def invertNumber(number):
    strnumber=str(number)
    if isOperator(strnumber[0]):
        strnumber.replace(strnumber[0],findInverse(strnumber[0]))
    else:
        strnumber="-"+strnumber
    return strnumber

def resolve(equation):
    ineq=getSigneOp(equation)
    equation2=""
    if(re.split(r'[<>=]',equation)[1]).__contains__("A"):
        equation2=equation
    else:
        equation2=arranger_l_equation(equation)
    letter="A"
    print("abbbbb ", equation2)
    splited_eq=equation.split("=")
    print("abbbbb ",splited_eq)
    res=[]
    for i in splited_eq:
        res.append(calculate(i))
    print(f">?? {res}")
    gauche_x=""
    droite_sans_x=""
    # gauche
    for i in res[0]:
        if str(i).__contains__("x"):
            u = addSign(i)
            gauche_x+=u
        else:
            droite_sans_x+=(invertNumber(i))

    #         droite
    for i in res[1]:
        if str(i).__contains__("x"):
            gauche_x += (invertNumber(i))
        else:
            u=addSign(i)
            droite_sans_x += str(u)

    equation2=gauche_x+"="+droite_sans_x
    splited_eq = equation2.split("=")
    res = []
    for i in splited_eq:
        res.append(calculate(i))


    # dernier rearrangement
    strs="x="+(res[1][0])+"/"+(res[0][0]).split("x")[0]
    result=letter+""+ineq+""+str(calculate(strs.split("=")[1])[0])
    return result

def addSign(number):
    u=number
    if isOperator(str(number)[0]) == False:
        u = "+" + number
    return u

def createCourbe(equation):
    operateur=getSigneOp(equation)
    t=equation.split(operateur)
    x = np.linspace(-10, 10, 100)
    y = 5*x+8
    y2 = 51*x-6
    plt.plot(x, y, label=y)
    plt.plot(x, y2, label=y2)
    if(operateur=="<"):
        plt.fill_between(x, y, y2, where=(y < y2), color='gray', alpha=0.3)
    elif (operateur==">") :
        plt.fill_between(x, y, y2, where=(y < y2), color='gray', alpha=0.3)
        # Ajouter des labels aux axes
        plt.xlabel('x')
        plt.ylabel('y')

        # Ajouter une légende
        plt.legend()

        # Enregistrer la courbe dans un fichier temporaire
        graph_file = '/Users/priscafehiarisoadama/me/S4/optimisation/TraitementImages/static/courbe.png'  # Choisissez le chemin approprié dans votre projet Flask
        plt.savefig(graph_file)

    print(y,y2)









if __name__ == '__main__':
    equation="x+3x=((9-3+4)/2)"
    eq=equation.split("=")
    e=[]
    for i in eq:
        e.append(calculate(i))

    for i in e:
        print(i)
    e="-11x+23"
    # r=resolution(e)
    # w=distribution_de_l_equation(r)
    # print(w)
    # print(findInverse("-"))

    equation="1-6x=6"
    # createCourbe(equation)
    print(f">>>>>>>{resolve(equation)}")
    # des=equation.split("=")
    # w=calculate(des[0])
    # print(f"w: {w}")
    # q=calculate(des[1])
    # print(f"q: {q}")

