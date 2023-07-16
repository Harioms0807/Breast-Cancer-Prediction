from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
import streamlit as st
import pandas as pd
import joblib
import json

@st.cache_resource
def load_resources():
    model = joblib.load(r'Datasets and Models/classifier.pkl')
    scaler = joblib.load(r'Datasets and Models/scaler.pkl')
    return model, scaler

def scale_data(data, scaler):
    scaled_data = scaler.transform(data)
    return scaled_data

def predict_cancer(data, model):
    predictions = model.predict(data)
    return predictions

def display_results(predictions):
    if predictions == 'M':
        st.success("The tumor cell seems to be malignant and the patient needs to be treated as soon as possible.")
        st.warning("Be cautious as this is just a predictive model and doesn't always reflect the ground scenario, although the model has high accuracy. Please consult a specialized doctor for further evaluation.")
    else:
        st.success("The tumor cell seems to be benign and the patient does not require any cancer treatment.")
        st.warning("Be cautious as this is just a predictive model and doesn't always reflect the ground scenario, although the model has high accuracy. Please consult a specialized doctor for further evaluation.")

def get_description(column):
    descriptions = {
        'radius_mean': 'Mean of distances from center to points on the perimeter',
        'texture_mean': 'Mean gray-scale value',
        'perimeter_mean': 'Perimeter of the tumor',
        'area_mean': 'Area of the tumor',
        'smoothness_mean': 'Local variation in radius lengths',
        'compactness_mean': 'Perimeter^2 / area - 1.0',
        'concavity_mean': 'Severity of concave portions of the contour',
        'concave points_mean': 'Number of concave portions of the contour',
        'symmetry_mean': 'Symmetry of the tumor',
        'fractal_dimension_mean': 'Fractal dimension of the tumor',
        'radius_se': 'Standard error of distances from center to points on the perimeter',
        'texture_se': 'Standard deviation of gray-scale values',
        'perimeter_se': 'Standard error of the tumor perimeter',
        'area_se': 'Standard error of the tumor area',
        'smoothness_se': 'Standard error of local variation in radius lengths',
        'compactness_se': 'Standard error of perimeter^2 / area - 1.0',
        'concavity_se': 'Standard error of severity of concave portions of the contour',
        'concave points_se': 'Standard error of number of concave portions of the contour',
        'symmetry_se': 'Standard error of symmetry of the tumor',
        'fractal_dimension_se': 'Standard error of fractal dimension of the tumor',
        'radius_worst': 'Worst (largest) value of distances from center to points on the perimeter',
        'texture_worst': 'Worst (largest) value of gray-scale values',
        'perimeter_worst': 'Worst (largest) value of the tumor perimeter',
        'area_worst': 'Worst (largest) value of the tumor area',
        'smoothness_worst': 'Worst (largest) value of local variation in radius lengths',
        'compactness_worst': 'Worst (largest) value of perimeter^2 / area - 1.0',
        'concavity_worst': 'Worst (largest) value of severity of concave portions of the contour',
        'concave points_worst': 'Worst (largest) value of number of concave portions of the contour',
        'symmetry_worst': 'Worst (largest) value of the tumor symmetry',
        'fractal_dimension_worst': 'Worst (largest) value of the fractal dimension of the tumor'
    }
    return descriptions.get(column, column)


def main():
    
    st.set_page_config("Breast Cancer Prediction", r'Related Images and Videos/breast.png', layout="wide")
    
    st.markdown(
        """
        <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
    
    page_title, lottie, buff= st.columns([65, 25, 5])
    
    page_title.title('Breast Cancer Prediction')
    
    with open (r"Related Images and Videos/breast.json") as f:
        lottie_json = json.load(f)
    with lottie:
        st_lottie(lottie_json, height= 100, width=200)
        
    user_input_checkbox = st.checkbox("Enter Your Own Values")
    
    add_vertical_space(2)
    
    X_test = pd.read_csv(r'Datasets and Models/X_test.csv')
    model, scaler = load_resources()
    
    if user_input_checkbox:
        
        input_data = pd.DataFrame(columns=X_test.columns)
        
        num_features = len(X_test.columns)
        num_columns = 3
        
        input_cols = st.columns(num_columns)
        
        input_values = []
        for i in range(0, num_features, num_columns):
            for j in range(num_columns):
                if i + j < num_features:
                    column = X_test.columns[i + j]
                    description = get_description(column)
                    value = input_cols[j].text_input(column.replace('_', ' - ').title().replace('Se', 'Standard Error').replace('Fractal - Dimension', 'Fractal Dimension'), help = description, value='')
                    input_values.append(value)
        
        add_vertical_space(1)   
        predict_button = st.button("Predict")
        
        if predict_button:
            if len(input_values) == num_features:
                input_data.loc[0] = input_values
                scaled_data = scale_data(input_data, scaler)
                predictions = predict_cancer(scaled_data, model)
                display_results(predictions[0])
            else:
                st.write("Please enter values for all features.")
    else:
        
        st.write("Select Values From Test Data:")
        
        add_vertical_space(1)
        
        select_data = pd.DataFrame(columns=X_test.columns)
        
        num_features = len(X_test.columns)
        num_columns = 3
        
        select_cols = st.columns(num_columns)
        
        selected_values = []
        for i in range(0, num_features, num_columns):
            for j in range(num_columns):
                if i + j < num_features:
                    column = X_test.columns[i + j]
                    values = X_test[column].unique()
                    description = get_description(column)
                    value = select_cols[j].selectbox(column.replace('_', ' - ').title().replace('Se', 'Standard Error').replace('Fractal - Dimension', 'Fractal Dimension'), values, help = description, index=0)
                    selected_values.append(value)
                
        add_vertical_space(1)
        predict_button = st.button("Predict")
        
        if predict_button:
            if len(selected_values) == num_features:
                select_data.loc[0] = selected_values
                scaled_data = scale_data(select_data, scaler)
                predictions = predict_cancer(scaled_data, model)
                display_results(predictions[0])
            else:
                st.write("Please select values for all features.")
    
    with st.sidebar:
    
        st.image(r'Related Images and Videos/side.png', use_column_width=True)
        st.title("About")
        st.write("This is a Breast Cancer Prediction App powered by SVC model. The app takes various features of a breast tumor as input and predicts whether the tumor is benign or malignant.")
        st.markdown("- Accuracy: 97.36%")
        st.markdown("- False Negatives: 0.87%")
        
        st.title("Performance Metrics")
        metrics_data = {
            'Tumor Type': ['Benign', 'Malignant'],
            'Precision': ['98%', '96%'],
            'Recall': ['97%', '98%'],
            'F1-Score': ['98%', '97%']
        }
        metrics = pd.DataFrame(metrics_data)
        st.dataframe(metrics, hide_index=True)


if __name__ == '__main__':
    main()
