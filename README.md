# CAT-3-Data-Science
# **Scenario 2: Student Enrollment Prediction**

## Objective and Data
The goal is to develop a **classification model** to predict:
1.  Which admitted students are likely to **enroll** in a program.
2.  Which enrolled students may need **additional support to graduate**.

The model is to be built using **historical student enrollment data**, **academic records**, and **demographic data**.(My data is in the code itself)

## Guiding Questions: Answers

1.  **Which type of machine learning algorithm would be most suitable for this task? Explain your reasoning.**
    * **Answer:** **Classification algorithms**, such as **Random Forest** or **Logistic Regression**, are most suitable.
    * **Reasoning:** Both enrollment (Enroll/Not Enroll) and graduation (Graduate/Not Graduate) are **binary, discrete outcomes**. Since the historical data includes the known outcome, this is a **supervised classification** problem. Random Forest is often preferred for its robustness and ability to provide **feature importance**.

2.  **Which features from the student data would be most relevant for predicting enrollment and graduation success?**
    * **Answer:**
        * **Enrollment Prediction:** **Financial Aid Amount** (often the top driver), **High School GPA**, **SAT/ACT Scores**, **Program of Choice**, and geographic distance/region.
        * **Graduation Success:** **First-Year Course Performance**, **High School GPA**, **Engagement with Support Services** (e.g., tutoring), and **First-Generation status** (as an indicator of required support).

3.  **How can you protect the privacy of student data while still using it to develop predictive models?**
    * **Answer:** Through **De-identification** (removing names, IDs), **Pseudonymization** (replacing identifiers with synthetic tokens), **Data Aggregation** (using ranges instead of exact sensitive numbers like income), and employing strong **Encryption** for data at rest and in transit, ensuring strict compliance with regulatory standards like **FERPA**.

4.  **How can you communicate the results of your model to educational institutions in a way that is actionable and informative?**
    * **Answer:** The results must be translated from technical metrics into **actionable strategies**:
        * **Risk Tiers:** Categorize students into **High, Medium, and Low-Risk** groups for non-enrollment/non-graduation.
        * **Policy Guidance:** Use **Feature Importance scores** to show which institutional levers (e.g., increasing financial aid, expanding tutoring) will have the greatest impact.
        * **Visualization:** Use clear dashboards and charts to communicate insights to non-technical staff.

  AUTHOR SOPHY NALIAKA BSE-05-0183/2024
