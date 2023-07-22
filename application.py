from flask import Flask,request,render_template
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application
#progress

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET', 'POST'])
def predictdata():
    if request.method =='POST':
        data = CustomData(
        cement= float(request.form.get('cement')),
        blast_furnace_slag= float(request.form.get('blast_furnace_slag')) ,
        fly_ash =float(request.form.get('fly_ash')) ,
        water = float(request.form.get('water')) ,
        superplasticizer= float(request.form.get('superplasticizer')),
        coarse_aggregate = float(request.form.get('coarse_aggregate')) ,
        fine_aggregate= float(request.form.get('fine_aggregate')) ,
        age= request.form.get('age') 
        )

        predict_df = data.get_data_as_data_frame()
        #print(predict_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(predict_df)
        #print(results)
        final_result = "%.2f" % round(results[0], 2)
        return render_template('index.html', prediction_text=f"The Concrete compressive strength is {final_result} MPa")
      
    

if __name__=="__main__":       
    app.run(host="0.0.0.0",port=4000)
