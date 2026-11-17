# Office IEQ Comfort Assessment — setup

## Windows installation

1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Open the extracted folder.
4. Click the File Explorer address bar, type `cmd`, press Enter.
5. Run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is not recognized, try `py` instead of `python`.

6. Start:

```bash
python -m streamlit run app.py
```

7. Open the local URL printed in Command Prompt (normally http://localhost:8501).

## Files
- `app.py`: Streamlit application and Random Forest training/inference code.
- `office_ieq_synthetic_training_data.csv`: the actual synthetic dataset loaded and used by the model.
- `requirements.txt`: Python package list.

Keep the CSV in the same folder as `app.py`.

## Important research limitations
The CSV contains generated synthetic examples, not measured people or a measured office. Its labels are generated using transparent demonstration assumptions. A high test accuracy only indicates that the model learned those synthetic assumptions; it is not evidence of real-world validity.

The app calculates PMV/PPD separately with `pythermalcomfort` using ISO 7730:2025 model selection. PMV/PPD are model estimates for a standardized occupant/group and do not replace an individual's thermal sensation vote. Some input combinations fall outside model applicability limits; in that case the app reports that PMV/PPD could not be calculated rather than inventing a result.

Clothing presets are approximate. Use measured/justified clothing ensemble insulation values where available. Robot noise/proximity links are simulated hypotheses until real robot and occupant data are collected.
