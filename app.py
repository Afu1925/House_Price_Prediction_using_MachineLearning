from flask import Flask,render_template,request,redirect,url_for,make_response,session
from flask_mysqldb import MySQL
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

app=Flask(__name__)
app.secret_key = '#&$44'
model = joblib.load(r"D:/house_price_prediction/Model/chennai_house_price_prediction_model.sav")
print(type(model))


        
def plot_correlation_heatmap(df):
    correlation = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig('static/correlation_heatmap.png')
    plt.close()

def plot_sales_price_distribution(sales_price_series):
    unique_sales_price = sales_price_series.value_counts().sort_index()
    unique_sales_price.head(20).plot(kind='bar', color='mediumseagreen')
    plt.title('Number of Properties by Sales Price')
    plt.xlabel('Sales Price')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('static/sales_price_distribution.png')
    plt.close()

def plot_model_evaluation_chart(results):
    models = list(results.keys())
    r2_scores = [metrics['R2'] for metrics in results.values()]
    mae_scores = [metrics['MAE'] for metrics in results.values()]
    mse_scores = [metrics['MSE'] for metrics in results.values()]

    x = np.arange(len(models))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, r2_scores, color='dodgerblue')
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.ylim(0, 1.1)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha='center')
    plt.tight_layout()
    plt.savefig('static/r2_scores.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, mae_scores, color='mediumseagreen')
    plt.title('MAE Comparison')
    plt.ylabel('Mean Absolute Error')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000, f"{bar.get_height():,.0f}", ha='center')
    plt.tight_layout()
    plt.savefig('static/mae_chart.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, mse_scores, color='salmon')
    plt.title('MSE Comparison')
    plt.ylabel('Mean Squared Error')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000, f"{bar.get_height():,.0f}", ha='center')
    plt.tight_layout()
    plt.savefig('static/mse_chart.png')
    plt.close()


app.config["MYSQL_HOST"]="localhost"
app.config["MYSQL_USER"]="root"
app.config["MYSQL_PASSWORD"]="tashfeen"
app.config["MYSQL_DB"]="signup"
app.config["MYSQL_CURSORCLASS"]="DictCursor"
mysql=MySQL(app)

@app.route('/')
def welcome():
 return render_template('welcome.html')

@app.route('/register',methods=['GET','POST'])
def register():
 if request.method=='POST':
        name=request.form['name']
        email=request.form['email']
        password=request.form['password']
        phone=request.form['phone']
        con=mysql.connection.cursor()
        sql="insert into user(name,email,password,phone) values(%s,%s,%s,%s)"
        con.execute(sql,(name,email,password,phone))
        mysql.connection.commit()
        con.close()
        return redirect(url_for('register'))
 return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        passw = request.form['password']
        con = mysql.connection.cursor()
        sql = "SELECT * FROM user WHERE email = %s AND password = %s"
        con.execute(sql, (email,passw))
        user = con.fetchone()
        con.close()
        if user:
           session['email']= email
           resp=redirect(url_for('predict'))
           resp.set_cookie('email',email, max_age=60*60*24)
           return resp
        
        else:
           return "Please check your email id and password"
        
    return render_template('login.html')

@app.route('/home1', methods=["GET", "POST"])
def predict():
    if 'email' not in session:
        return redirect(url_for('login'))
    if request.method == "POST":
        try:
            data = {
                "PRT_ID": ["Dummy"],
                "AREA": [request.form["area"]],
                "INT_SQFT": [float(request.form["int_sqft"])],
                "DIST_MAINROAD": [float(request.form["dist_mainroad"])],
                "N_BEDROOM": [int(request.form["n_bedroom"])],
                "N_BATHROOM": [int(request.form["n_bathroom"])],
                "N_ROOM": [int(request.form["n_room"])],
                "SALE_COND": [request.form["sale_cond"]],
                "PARK_FACIL": [request.form["park_facil"]],
                "BUILDTYPE": [request.form["buildtype"]],
                "UTILITY_AVAIL": [request.form["utility_avail"]],
                "STREET": [request.form["street"]],
                "MZZONE": [request.form["mzzone"]],
                "QS_ROOMS": [float(request.form["qs_rooms"])],
                "QS_BATHROOM": [float(request.form["qs_bathroom"])],
                "QS_BEDROOM": [float(request.form["qs_bedroom"])],
                "QS_OVERALL": [float(request.form["qs_overall"])],
                "COMMIS": [float(request.form["commis"])]
            }
            input_df = pd.DataFrame(data)

            predicted_price = model.predict(input_df)[0]

            return render_template("home1.html", prediction=f"Predicted House Price: ₹{predicted_price:,.2f}")

        except Exception as e:
            return render_template("home1.html", error=f"Error: {e}")

    return render_template("home1.html")

@app.route('/admin_login', methods=["GET", "POST"])
def admin_login():
    if request.method == 'POST':
        admin_name = request.form['admin_name']
        pass_adm = request.form['pass_adm']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM admin WHERE admin_name = %s AND pass_adm = %s", (admin_name, pass_adm))
        result = cursor.fetchone()
        cursor.close()

        if result and admin_name == 'Afreen' and pass_adm == '123':
           session['admin_logged_in'] = True
           session['admin_name'] = admin_name
           
           return redirect(url_for('editing'))
        else:
            return "Please check your username and password"
    return render_template('admin_login.html')

@app.route('/confirm')
def confirm():
    return render_template('confirm.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/editing', methods=['GET', 'POST'])
def editing():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    try:
        if request.method == 'POST':
            prt_id=request.form['prt_id']
            area = request.form['area']
            int_sqft = float(request.form['int_sqft'])
            dist_mainroad = float(request.form['dist_mainroad'])
            n_bedroom = int(request.form['n_bedroom'])
            n_bathroom = int(request.form['n_bathroom'])
            n_room = int(request.form['n_room'])
            sale_cond = request.form['sale_cond']
            park_facil = request.form['park_facil']
            buildtype = request.form['buildtype']
            utility_avail = request.form['utility_avail']
            street = request.form['street']
            mzzone = request.form['mzzone']
            qs_rooms = float(request.form['qs_rooms'])
            qs_bathroom = float(request.form['qs_bathroom'])
            qs_bedroom = float(request.form['qs_bedroom'])
            qs_overall = float(request.form['qs_overall'])
            commis = float(request.form['commis'])
            sales_price=int(request.form['sales_price'])

            df = pd.read_csv(r'D:/chennai_house_price_prediction.csv')
            new_data = {
                'PRT_ID':prt_id,
                'AREA': area,
                'INT_SQFT': int_sqft,
                'DIST_MAINROAD': dist_mainroad,
                'N_BEDROOM': n_bedroom,
                'N_BATHROOM': n_bathroom,
                'N_ROOM': n_room,
                'SALE_COND': sale_cond,
                'PARK_FACIL': park_facil,
                'BUILDTYPE': buildtype,
                'UTILITY_AVAIL': utility_avail,
                'STREET': street,
                'MZZONE': mzzone,
                'QS_ROOMS': qs_rooms,
                'QS_BATHROOM': qs_bathroom,
                'QS_BEDROOM': qs_bedroom,
                'QS_OVERALL': qs_overall,
                'COMMIS': commis,
                'SALES_PRICE': sales_price
            }
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(r'D:/chennai_house_price_prediction.csv', index=False)
    except Exception as e:
        return render_template('editing.html', error=f"Error occurred while adding data: {e}")

    try:
        df = pd.read_csv(r'D:/chennai_house_price_prediction.csv')
        last_five = df.tail(5)
        return render_template('editing.html', houses=last_five.to_dict(orient='records'))
    except Exception as e:
        return render_template('editing.html', error=f"Error loading dataset: {e}")

@app.route('/right')
def right():
    df = pd.read_csv('D:/chennai_house_price_prediction.csv')  

    plot_correlation_heatmap(df)
    plot_sales_price_distribution(df['SALES_PRICE'])
    with open(r'D:/house_price_prediction/Model/evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
    results = {
        'LinearRegression': {'R2': 0.87, 'MAE': 1060522.56, 'MSE': 1731518311073.76},
        'DecisionTreeRegressor':{'R2': 0.96, 'MAE': 594602.21, 'MSE': 587210673635.32},
        'RandomForestRegressor': {'R2': 0.98, 'MAE': 402347.95, 'MSE': 269716971196.65},
        'GradientBoostingRegressor': {'R2': 0.99, 'MAE': 342257.48, 'MSE': 192272218101.72}
    }
    plot_model_evaluation_chart(results)

    return render_template('right.html', metrics=metrics) 

@app.route('/dataset')
def dataset():
    df = pd.read_csv('D:/chennai_house_price_prediction.csv')

    if 'PRT_ID' in df.columns:
        df = df.drop(columns=['PRT_ID'])
    
    if 'AREA' in df.columns:
        df['AREA']=df['AREA'].replace(
        {
            'Chrompt':'Chrompet',
            'Chrmpet':'Chrompet',
            'Chormpet':'Chrompet',
            'Karapakam':'Karapakkam',
            'KKNagar':'KK Nagar',
            'Velchery':'Velachery',
            'Ana Nagar':'Anna Nagar',
            'Ann Nagar':'Anna Nagar',
            'Adyr':'Adyar',
            'TNagar':'T Nagar'
        }
        )
        df['AREA']=df['AREA'].str.title()
    if 'SALE_COND' in df.columns:
        df['SALE_COND']=df['SALE_COND'].replace(
        {
            'Adj Land':'AdjLand',
            'Partiall':'Partial',
            'PartiaLl':'Partial',
            'Ab Normal':'AbNormal'
        }
        )
        df['SALE_COND']=df['SALE_COND'].str.title()
    if 'PARK_FACIL' in df.columns:
        df['PARK_FACIL']=df['PARK_FACIL'].replace(
        {
            'Noo':'No'
        }
        )
        df['PARK_FACIL']=df['PARK_FACIL'].str.title()
    if 'BUILDTYPE' in df.columns:
        df['BUILDTYPE']=df['BUILDTYPE'].replace(
        {
            'Comercial':'Commercial',
            'Other':'Others'
        }
        )
        df['BUILDTYPE']=df['BUILDTYPE'].str.title()
    if 'UTILITY_AVAIL' in df.columns:
        df['UTILITY_AVAIL']=df['UTILITY_AVAIL'].replace(
        {
            'All Pub':'AllPub',
            'NoSewr':'NoSeWa'
        }
        )
        df['UTILITY_AVAIL']=df['UTILITY_AVAIL'].str.title()
    if 'STREET' in df.columns:
        df['STREET']=df['STREET'].replace(
        {
            'Pavd':'Paved',
            'NoAccess':'No Access'
        }
        )
        df['STREET']=df['STREET'].str.title()
        df['N_BEDROOM'].fillna(df['N_BEDROOM'].mode()[0], inplace=True)
        df['N_BATHROOM'].fillna(df['N_BATHROOM'].mode()[0], inplace=True)
        df['QS_OVERALL'].fillna(df['QS_OVERALL'].mean(), inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    categorical_data = []
    numerical_data = []

    for col in categorical_cols:
        unique_vals = df[col].value_counts(dropna=False).to_dict()
        total_vals = df[col].count()
        categorical_data.append({
            'name': col,
            'unique_values': unique_vals,
            'total_values': total_vals
        })

    numerical_data = []
    for col in numerical_cols:
        total_vals = df[col].count()
        unique_count = df[col].nunique(dropna=False)
        numerical_data.append({
            'name': col,
            'total_values': total_vals,
            'unique_count': unique_count
        })
    return render_template('dataset.html',
                           categorical_data=categorical_data,
                           numerical_data=numerical_data)

@app.route('/pred_dash')
def pred_dash():
    df = pd.read_csv('D:/chennai_house_price_prediction.csv') 

    plot_correlation_heatmap(df)
    plot_sales_price_distribution(df['SALES_PRICE'])
    with open(r'D:/house_price_prediction/Model/evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
    results = {
        'LinearRegression': {'R2': 0.87, 'MAE': 1054360.87, 'MSE': 1712501800441.39},
        'DecisionTreeRegressor':{'R2': 0.95, 'MAE': 586545.21, 'MSE': 627496813402.74},
        'RandomForestRegressor': {'R2': 0.98, 'MAE': 408025.89, 'MSE': 297774133499.47},
        'GradientBoostingRegressor': {'R2': 0.98, 'MAE': 367622.50, 'MSE': 246435904877.64}
    }
    plot_model_evaluation_chart(results)
    return render_template('pred_dash.html',metrics=metrics)

@app.route('/left')
def left():
    return render_template('left.html')

@app.route('/contact',methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        feedback = request.form['feedback']
        conn=mysql.connection.cursor()
        sql="insert into contact(name,email,feedback) values(%s,%s,%s)"
        conn.execute(sql,(name,email,feedback))
        mysql.connection.commit()
        conn.close()
        welcomename=name=request.cookies.get('name')
        return render_template('contact.html',name=welcomename,message="Thank you for your feedback!")

    return render_template('contact.html')
@app.route('/services')
def services():
    return render_template('services.html')
@app.route('/logout')
def logout():
    session.clear()  
    resp = make_response(redirect(url_for('login')))
    return resp
@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')
@app.route('/distribution')
def distribution():
    return render_template('distribution.html')
@app.route('/r2scores')
def r2scores():
    return render_template('r2scores.html')
@app.route('/mae_chart')
def mae_chart():
    return render_template('mae_chart.html')
@app.route('/mse_chart')
def mse_chart():
    return render_template('mse_chart.html')
if __name__ == '__main__':
    app.run(debug=True)