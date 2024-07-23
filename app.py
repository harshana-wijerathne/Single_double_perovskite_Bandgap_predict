from flask import Flask, render_template, request,redirect,url_for
import pickle
import math
import numpy as np

app = Flask(__name__)

def prediction(lst):
    try:
        filename = 'model/Bandgap_predicter_model.pickle'  # Use forward slash for path
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        pred_value1 = model.predict([lst])
        pred_value =(pred_value1*0.7911172826831386)+0.48805921634099875
        return pred_value
    except Exception as e:
        return str(e)

def t(a, b, x, n):
    try:
        t_value = (a + x) / ((2 ** 0.5) * (b + x))
        return t_value
    except Exception as e:
        return str(e)

def tau(a, b, x, n):
    try:
        tau_value = (x / b) - n * (n - (a / b) / (math.log(a / b, math.e)))
        return tau_value
    except Exception as e:
        return 999
    
def d_prediction(lst):
    try:
        filename = 'model/DoublePerovskiteBandgap.pickle'  # Use forward slash for path
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        pred_value = model.predict([lst])
        return pred_value
    except Exception as e:
        return str(e)

def d_t(a, b1,b2, x, n):
    try:
        b=(b1+b2)/2
        t_value = (a + x) / ((2 ** 0.5) * (b + x))
        return t_value
    except Exception as e:
        return str(e)

def d_tau(a, b1,b2, x, n):
    try:
        b=(b1+b2)/2
        tau_value = (x / b) - n * (n - (a / b) / (math.log(a / b, math.e)))
        return tau_value
    except Exception as e:
        return 999

@app.route('/', methods=["POST", "GET"])
def index():
    pred1 = 0
    tau11 = 0
    t11 = 0
    formula1 = ''
    
    if request.method == 'POST':
        try:
            eleA1 = request.form["radiusA1"]
            eleB1 = request.form["radiusB1"]
            eleX1 = request.form["radiusC1"]

            A = ['Ag', 'Ba', 'Cs', 'In', 'K', 'Li', 'Rb', 'Tl', 'MA', 'FA']
            rA = [1.15, 1.64569, 1.88, 0.93412, 1.64, 1.18, 1.72, 1.7, 2.71, 2.85]
            nA = [1, 2, 1, 3, 1, 1, 1, 1, 1, 1]
            Ea = [7.57, 9.98, 3.89, 28.14, 4.34, 5.4, 2.52, 6.11, 9.3, 9.0]

            B = ['Ag','Al', 'As', 'Au', 'Ba', 'Be', 'Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 'Hg', 'In', 'Ir', 'Li', 'Mg', 'Mn', 'Na', 'Ni', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rh', 'Ru', 'Sb', 'Sc', 'Se', 'Si', 'Sn', 'Sr', 'Ti', 'Tl', 'V', 'Y', 'Zn', 'Zr']
            rB = [1.15,0.54, 0.46, 1.37, 1.61, 0.49, 1.0, 0.95, 0.74, 0.8, 0.73, 0.78, 0.62, 0.73, 0.71, 1.02, 0.8, 0.59, 0.72, 0.72, 0.83, 1.02, 0.69, 0.71, 1.54, 1.19, 0.86, 0.77, 0.55, 0.68, 0.62, 0.75, 0.64, 0.4, 1.15, 1.18, 0.84, 1.7, 0.79, 0.9, 0.74, 0.72]
            nB = [1,3, 5, 1, 2, 2, 2, 2, 3, 3, 2, 3, 3, 4, 4, 2, 3, 3, 1, 2, 3, 1, 2, 4, 5, 2, 2, 2, 3, 3, 3, 3, 2, 4, 4, 2, 4, 2, 5, 2, 2, 2]
            Eb = [7.57,27.45, 62.73, 8.9, 9.65, 17.81, 11.87, 16.91, 32.45, 29.61, 20.29, 30.63, 30.7, 45.81, 31.26, 18.76, 28.6, 30.08, 5.39, 15.03, 33.43, 5.14, 18.17, 44.9, 65.12, 21.47, 19.52, 18.56, 31.06, 28.47, 25.56, 24.36, 21.2, 45.14, 40.73, 11.03, 40.98, 24.56, 65.23, 12.24, 17.96, 14.61]

            X = ['F', 'Cl', 'Br', 'I']
            rX = [1.33, 1.81, 1.96, 2.2]
            nX = [-1, -1, -1, -1]
            Ex = [-3.28, -3.61, -3.36, -3.06]

            if eleA1 not in A or eleB1 not in B or eleX1 not in X:
                raise KeyError("Invalid element input.")

            features1 = []
            features1.append(rA[A.index(eleA1)])
            features1.append(rB[B.index(eleB1)])
            features1.append(rX[X.index(eleX1)])
            features1.append(nA[A.index(eleA1)])
            features1.append(nB[B.index(eleB1)])
            features1.append(nX[X.index(eleX1)])
            features1.append(Ea[A.index(eleA1)])
            features1.append(Eb[B.index(eleB1)])
            features1.append(Ex[X.index(eleX1)])

            pred1 = prediction(features1)
            if isinstance(pred1, str):
                raise ValueError(pred1)
            

            tau11 = tau(features1[0], features1[1], features1[2], features1[3])
            if isinstance(tau11, str):
                raise ValueError(tau11)
            

            t11 = t(features1[0], features1[1], features1[2], features1[3])
            if isinstance(t11, str):
                raise ValueError(t11)
            if tau11 ==999:
                tau11=0
                t11=0
                pred1=0
                formula1 = "Not Perovskite"
            else:

                t11 = np.round(t11, 3)
                tau11 = np.round(tau11, 3)
                pred1 = np.round(pred1[0], 3)
                formula1 = eleA1 + eleB1 + eleX1

        except KeyError as e:
            return render_template("error.html", error_message=f"Invalid input: {e}")
        except ValueError as e:
            return render_template("error.html", error_message=str(e))
        except Exception as e:
            return render_template("error.html", error_message=f"Unexpected error: {e}")

    return render_template("index.html", pred_value1=pred1, tau_value1=tau11, t_value1=t11, formula1=formula1)

@app.route('/second_page', methods=["POST", "GET"])
def second_page():
    pred = 0
    tau1 = 0
    t1 = 0
    eleA=''
    eleB1=''
    eleB2=''
    eleX = ''
    
    if request.method == 'POST':
        try:
            eleA = request.form["radiusA"]
            eleB1 = request.form["radiusB1"]
            eleB2 = request.form["radiusB2"]
            eleX = request.form["radiusC"]

            A = ['Ag', 'Ba', 'Cs', 'In', 'K', 'Li', 'Rb', 'Tl', 'MA', 'FA']
            rA = [1.15, 1.64569, 1.88, 0.93412, 1.64, 1.18, 1.72, 1.7, 2.71, 2.85]
            nA = [1, 2, 1, 3, 1, 1, 1, 1, 1, 1]
            Ea = [7.57, 9.98, 3.89, 28.14, 4.34, 5.4, 2.52, 6.11, 9.3, 9.0]

            B = ['Ag','Al', 'As', 'Au', 'Ba', 'Be', 'Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 'Hg', 'In', 'Ir', 'Li', 'Mg', 'Mn', 'Na', 'Ni', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rh', 'Ru', 'Sb', 'Sc', 'Se', 'Si', 'Sn', 'Sr', 'Ti', 'Tl', 'V', 'Y', 'Zn', 'Zr']
            rB = [1.15,0.54, 0.46, 1.37, 1.61, 0.49, 1.0, 0.95, 0.74, 0.8, 0.73, 0.78, 0.62, 0.73, 0.71, 1.02, 0.8, 0.59, 0.72, 0.72, 0.83, 1.02, 0.69, 0.71, 1.54, 1.19, 0.86, 0.77, 0.55, 0.68, 0.62, 0.75, 0.64, 0.4, 1.15, 1.18, 0.84, 1.7, 0.79, 0.9, 0.74, 0.72]
            nB = [1,3, 5, 1, 2, 2, 2, 2, 3, 3, 2, 3, 3, 4, 4, 2, 3, 3, 1, 2, 3, 1, 2, 4, 5, 2, 2, 2, 3, 3, 3, 3, 2, 4, 4, 2, 4, 2, 5, 2, 2, 2]
            Eb = [7.57,27.45, 62.73, 8.9, 9.65, 17.81, 11.87, 16.91, 32.45, 29.61, 20.29, 30.63, 30.7, 45.81, 31.26, 18.76, 28.6, 30.08, 5.39, 15.03, 33.43, 5.14, 18.17, 44.9, 65.12, 21.47, 19.52, 18.56, 31.06, 28.47, 25.56, 24.36, 21.2, 45.14, 40.73, 11.03, 40.98, 24.56, 65.23, 12.24, 17.96, 14.61]

            X = ['F', 'Cl', 'Br', 'I']
            rX = [1.33, 1.81, 1.96, 2.2]
            nX = [-1, -1, -1, -1]
            Ex = [-3.28, -3.61, -3.36, -3.06]

            if eleA not in A or eleB1 not in B or eleB2 not in B or eleX not in X:
                raise KeyError("Invalid element input.")

            features = []
            features.append(rA[A.index(eleA)])
            features.append(rB[B.index(eleB1)])
            features.append(rB[B.index(eleB2)])
            features.append(rX[X.index(eleX)])
            features.append(nA[A.index(eleA)])
            features.append(nB[B.index(eleB1)])
            features.append(nB[B.index(eleB2)])
            features.append(nX[X.index(eleX)])
            features.append(Ea[A.index(eleA)])
            #features.append(Eb[B.index(eleB)])
            #features.append(Ex[X.index(eleX)])
            print(features)
            pred = d_prediction(features)
            if isinstance(pred, str):
                raise ValueError(pred)
            

            tau1 = d_tau(features[0], features[1], features[2], features[3],features[4])
            if isinstance(tau1, str):
                raise ValueError(tau1)
            

            t1 = d_t(features[0], features[1], features[2], features[3],features[4])
            if isinstance(t1, str):
                raise ValueError(t1)
            if tau1 ==999:
                tau1=0
                t1=0
                pred=0
                formula = "Not Perovskite"
            else:

                t1 = np.round(t1, 3)
                tau1 = np.round(tau1, 3)
                pred = np.round(pred[0], 3)
                #formula = eleA + eleB1 + eleB2 + eleX

        except KeyError as e:
            return render_template("error.html", error_message=f"Invalid input: {e}")
        except ValueError as e:
            return render_template("error.html", error_message=str(e))
        except Exception as e:
            return render_template("error.html", error_message=f"Unexpected error: {e}")

    return render_template("index2.html", pred_value=pred, tau_value=tau1, t_value=t1, a=eleA ,b1=eleB1, b2= eleB2, x= eleX)
    #return render_template('index2.html')

@app.route('/go_to_second_page', methods=['POST'])
def go_to_second_page():
    return redirect(url_for('second_page'))

@app.route('/go_to_first_page', methods=['POST'])
def go_to_first_page():
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
