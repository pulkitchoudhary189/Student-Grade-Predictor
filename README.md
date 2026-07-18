### 🎓 README for Student Grade Predictor
Create a file named `README.md` in your academic analytics directory and paste this:

```markdown
# Student Grade Predictor & Performance Analytics 🎓

A interactive machine learning dashboard designed to forecast final student grades and classify academic risk using supervised linear regression.

## 📝 Abstract
Proactive educational monitoring is essential for minimizing student attrition and optimizing academic pathways. This project presents a localized interactive dashboard engineered to predict a student’s final grade and classify their academic risk profile using predictive modeling. Built using Python, Scikit-learn, and Streamlit, the system functions via a Supervised Linear Regression pipeline. The underlying engine utilizes historical academic metrics—specifically focusing on attendance percentages, continuous assignment averages, and midterm examination scores—to train a predictive model. The script includes an automated data-generation mechanism that constructs a realistic dataset modeling real-world academic correlations. Data splits are rigorously maintained at an 80/20 train-test ratio to validate regression performance. The structural predictive accuracy is continuously evaluated using R-Squared ($R^2$) fit scores to track variance explanations alongside Mean Absolute Error (MAE) to quantify score deviation. The front-end interface provides real-time prediction matrix outputs, dynamically rendering conditional HTML status cards ("Excellent", "Clear", or "Critical Risk Profile") based on the regression results. This diagnostic environment empowers educators to manipulate specific performance variables via sliders to forecast academic outcomes instantly, demonstrating how transparent machine learning frameworks can serve as lightweight, accessible tools for institutional academic intervention strategies.

## 🚀 Key Features
* **Automated Data Safety-Net:** Built-in synthetic dataset generator that automatically provisions a realistic 250-student database (`student_grades.csv`) upon the first runtime.
* **Supervised Training Engine:** Utilizes Scikit-learn's `LinearRegression` pipeline with an 80/20 training/validation split.
* **Live Performance Metrics:** Displays real-time model evaluation indices ($R^2$ Fit Score and Mean Absolute Error percentage).
* **Conditional Interface Indicators:** Uses dynamic HTML formatting blocks to alert users if a student falls into an "Excellent", "Clear", or "Critical Risk" category.

## 📦 Installation & Setup
1. Clone this repository or open the project folder in your terminal.
2. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn
