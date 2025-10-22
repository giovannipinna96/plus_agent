#!/usr/bin/env python3
"""
10 Progressive Difficulty Questions for Titanic Dataset

These questions are designed to test the smolagents system's ability to:
- Use single and multiple tools
- Handle data transformations
- Perform statistical analysis
- Train and evaluate ML models
- Provide comprehensive answers

Run with: uv run python titanic_questions.py
"""

# ===========================================================================
# 10 DOMANDE DI DIFFICOLTÀ CRESCENTE SUL DATASET TITANIC
# ===========================================================================

TITANIC_QUESTIONS = [
    # LIVELLO 1: FACILE - Informazioni di base
    {
        "numero": 1,
        "livello": "FACILE",
        "domanda": "Give me some basic information about the Titanic dataset.",
        "tool_richiesti": [
            "get_dataset_insights()",
            "load_dataset()",
            "get_column_names()"
        ],
        "sequenza_tool_dettagliata": [
            {
                "step": 1,
                "tool": "load_dataset",
                "params": {"file_path": "data/titanic.csv"},
                "scopo": "Verificare che il dataset si carichi correttamente e ottenere dimensioni base"
            },
            {
                "step": 2,
                "tool": "get_dataset_insights",
                "params": {"file_path": "data/titanic.csv"},
                "scopo": "Ottenere overview completo con statistiche sopravvivenza, missing data, e patterns"
            },
            {
                "step": 3,
                "tool": "get_column_names",
                "params": {"file_path": "data/titanic.csv"},
                "scopo": "Lista dettagliata di tutte le colonne con categorizzazione (numeriche/categoriche)",
                "opzionale": True
            }
        ],
        "flusso_atteso": "Carica → Ottieni info di base → Mostra insights",
        "complessità": "Singolo tool o combinazione di 2-3 tool base",
        "output_atteso": """
        - Numero righe e colonne
        - Tasso di sopravvivenza generale
        - Colonne presenti (numeriche e categoriche)
        - Dati mancanti principali
        - Insights specifici per Titanic (sopravvivenza per genere/classe)
        """,
        "verifica_successo": [
            "Mostra numero di passeggeri (891)",
            "Indica tasso sopravvivenza (~38%)",
            "Lista colonne disponibili",
            "Identifica missing data (Cabin, Age)"
        ]
    },

    # LIVELLO 2: FACILE-MEDIO - Esplorazione semplice
    {
        "numero": 2,
        "livello": "FACILE-MEDIO",
        "domanda": "How many people survived and how many died?",
        "tool_richiesti": [
            "get_unique_values('survived')",
            "calculate_group_statistics('survived')"
        ],
        "flusso_atteso": "Carica → Conta sopravvissuti vs morti",
        "complessità": "Singolo tool, aggregazione semplice",
        "output_atteso": """
        - Numero totale sopravvissuti: 342 (38.4%)
        - Numero totale morti: 549 (61.6%)
        - Visualizzazione percentuali
        """,
        "verifica_successo": [
            "Mostra 342 sopravvissuti",
            "Mostra 549 morti",
            "Calcola percentuali corrette"
        ]
    },

    # LIVELLO 3: MEDIO - Analisi categorica
    {
        "numero": 3,
        "livello": "FACILE-MEDIO",
        "domanda": "What is the distribution of passengers by class (first, second, third)?",
        "tool_richiesti": [
            "get_unique_values('pclass')",
            "create_bar_chart('pclass')",
            "calculate_group_statistics('pclass')"
        ],
        "flusso_atteso": "Carica → Analizza distribuzione classi → Opzionale visualizzazione",
        "complessità": "1-2 tool, analisi categorica",
        "output_atteso": """
        - Prima classe: ~216 passeggeri (24%)
        - Seconda classe: ~184 passeggeri (21%)
        - Terza classe: ~491 passeggeri (55%)
        - Grafico a barre (opzionale)
        """,
        "verifica_successo": [
            "Conta corretta per ogni classe",
            "Percentuali accurate",
            "Identifica terza classe come più numerosa"
        ]
    },

    # LIVELLO 4: MEDIO - Analisi dati mancanti
    {
        "numero": 4,
        "livello": "MEDIO",
        "domanda": "Which columns have missing values and how many do they have?",
        "tool_richiesti": [
            "get_null_counts()",
            "fill_numeric_nulls('age')",  # potenziale
            "drop_null_rows()"  # potenziale
        ],
        "flusso_atteso": "Carica → Analizza missing → Raccomandazioni",
        "complessità": "Tool multipli, decision making",
        "output_atteso": """
        - Cabin: 687 mancanti (77.1%) → Considera eliminazione colonna
        - Age: 177 mancanti (19.9%) → Imputa con mediana/media
        - Embarked: 2 mancanti (0.2%) → Imputa con moda o elimina righe
        - Raccomandazioni specifiche per ciascuna
        """,
        "verifica_successo": [
            "Identifica tutte le colonne con missing",
            "Calcola percentuali corrette",
            "Fornisce raccomandazioni sensate basate su percentuale"
        ]
    },

    # LIVELLO 5: MEDIO - Analisi statistica numerica
    {
        "numero": 5,
        "livello": "MEDIO",
        "domanda": "Provide me with detailed statistics on passenger age. Were there any outliers?",
        "tool_richiesti": [
            "get_numeric_summary('age')",
            "create_histogram('age')"
        ],
        "flusso_atteso": "Carica → Calcola stats → Interpreta outlier → Opzionale viz",
        "complessità": "Interpretazione statistica, rilevamento outlier",
        "output_atteso": """
        - Media: ~29.7 anni
        - Mediana: ~28 anni
        - Range: 0.42 - 80 anni
        - Deviazione standard: ~14.5
        - Skewness: positiva (coda destra)
        - Outlier rilevati: età molto elevate (>65 anni)
        - 177 valori mancanti (19.9%)
        """,
        "verifica_successo": [
            "Calcola statistiche descrittive complete",
            "Identifica outlier con metodo IQR",
            "Interpreta distribuzione (skewness)",
            "Fornisce contesto (es. outlier sono validi)"
        ]
    },

    # LIVELLO 6: MEDIO-DIFFICILE - Correlazione
    {
        "numero": 6,
        "livello": "MEDIO-DIFFICILE",
        "domanda": "Is there a correlation between age and ticket price (fare)? Plot it.",
        "tool_richiesti": [
            "calculate_correlation('age', 'fare')",
            "create_scatter_plot('age', 'fare')"
        ],
        "flusso_atteso": "Carica → Calcola correlazione → Crea visualizzazione → Interpreta",
        "complessità": "Tool multipli, interpretazione statistica",
        "output_atteso": """
        - Coefficiente di correlazione di Pearson: ~0.09 (debole positiva)
        - P-value: significativo o non significativo
        - Scatter plot salvato
        - Interpretazione: correlazione debole/assente
        """,
        "verifica_successo": [
            "Calcola correlazione corretta",
            "Fornisce p-value per significatività",
            "Crea scatter plot",
            "Interpreta correttamente (correlazione debole)"
        ]
    },

    # LIVELLO 7: MEDIO-DIFFICILE - Confronto gruppi
    {
        "numero": 7,
        "livello": "MEDIO-DIFFICILE",
        "domanda": "Was survival different between men and women? Perform a statistical test.",
        "tool_richiesti": [
            "calculate_group_statistics('survived', 'sex')",
            "perform_ttest('survived', 'sex')",  # o chi_square_test
            "chi_square_test('survived', 'sex')"
        ],
        "flusso_atteso": "Carica → Stats per gruppo → Test statistico → Interpreta p-value",
        "complessità": "Test statistico, interpretazione significatività",
        "output_atteso": """
        - Sopravvivenza femmine: ~74.2%
        - Sopravvivenza maschi: ~18.9%
        - Test chi-quadrato: p < 0.001 (altamente significativo)
        - Conclusione: differenza statisticamente significativa
        - Donne avevano 3.9x più probabilità di sopravvivere
        """,
        "verifica_successo": [
            "Calcola tassi sopravvivenza per genere",
            "Esegue test appropriato (chi-quadrato per categorici)",
            "Interpreta p-value correttamente",
            "Conclude che differenza è significativa"
        ]
    },

    # LIVELLO 8: DIFFICILE - Pipeline ML completa
    {
        "numero": 8,
        "livello": "DIFFICILE",
        "domanda": "Create a machine learning model to predict survival. Which features are most important?",
        "tool_richiesti": [
            "get_null_counts()",  # identificare missing
            "fill_numeric_nulls('age')",  # o drop_null_rows
            "encode_categorical('sex')",
            "encode_categorical('embarked')",  # se usato
            "train_random_forest_model()",  # o train_logistic_regression
            "feature_selection()"  # o dalla feature importance del modello
        ],
        "flusso_atteso": "Carica → Gestisci missing → Encode categoriche → Addestra modello → Analizza importance",
        "complessità": "Pipeline ML completa, preprocessing multipli",
        "output_atteso": """
        - Modello addestrato con accuracy ~80-85%
        - Feature importance:
          1. Sex (~32%)
          2. Pclass (~28%)
          3. Fare (~19%)
          4. Age (~15%)
          5. Family size (~6%)
        - Metriche: precision, recall, F1-score
        - Cross-validation score
        """,
        "verifica_successo": [
            "Gestisce missing values appropriatamente",
            "Codifica variabili categoriche",
            "Addestra modello con performance ragionevole (>75%)",
            "Estrae e mostra feature importance",
            "Interpreta quali fattori contavano di più"
        ]
    },

    # LIVELLO 9: DIFFICILE - Analisi condizionale complessa
    {
        "numero": 9,
        "livello": "DIFFICILE",
        "domanda": "Among third-class passengers aged between 20 and 30, what percentage survived? Compare this with other classes.",
        "tool_richiesti": [
            "filter_rows_categorical('pclass', '3')",
            "filter_rows_numeric('age', '>=', 20)",
            "filter_rows_numeric('age', '<=', 30)",
            "calculate_group_statistics('survived', 'pclass')"
        ],
        "flusso_atteso": "Carica → Filtra multiplo → Calcola stats → Confronta gruppi",
        "complessità": "Operazioni sequenziali multiple, confronti",
        "output_atteso": """
        - Terza classe, 20-30 anni: ~120 passeggeri, ~30% sopravvissuti
        - Confronto con:
          * Prima classe, 20-30: ~40 passeggeri, ~60% sopravvissuti
          * Seconda classe, 20-30: ~35 passeggeri, ~45% sopravvissuti
        - Analisi: chiaro vantaggio classi superiori
        """,
        "verifica_successo": [
            "Applica filtri corretti in sequenza",
            "Calcola percentuali accurate",
            "Confronta across classi",
            "Identifica pattern socio-economico"
        ]
    },

    # LIVELLO 10: MOLTO DIFFICILE - Predizione personalizzata
    {
        "numero": 10,
        "livello": "MOLTO DIFFICILE",
        "domanda": "I am a 20-year-old male passenger on the Titanic, travelling first class. Calculate my probability of surviving the disaster.",
        "tool_richiesti": [
            "predict_single_passenger_survival(file_path='data/titanic.csv', passenger_age=20, passenger_sex='male', passenger_class=1)",
            # ALTERNATIVA con tool esistenti:
            "fill_numeric_nulls('age')",
            "encode_categorical('sex')",
            "train_random_forest_model('survived', 'pclass,sex,age,fare')",
            "make_prediction()"
        ],
        "flusso_atteso": "Carica → Prepara dati → Addestra modello → Predici → Spiega → Confronta storico",
        "complessità": "Workflow ML completo, predizione custom, spiegazione comprensiva",
        "output_atteso": """
        - Probabilità sopravvivenza: ~23-37% (varia leggermente per modello)
        - Predizione binaria: probabilmente NON sopravvissuto
        - Spiegazione factors:
          * Maschio: grande svantaggio (solo 19% uomini sopravvissuti)
          * Prima classe: vantaggio (37% maschi prima classe sopravvissuti)
          * Età 20: neutro/lieve vantaggio (giovane adulto)
          * Solo: lieve svantaggio
        - Confronto con passeggeri simili
        - Feature importance
        - Confidence level del modello
        """,
        "verifica_successo": [
            "Crea predizione personalizzata accurata",
            "Fornisce probabilità numerica",
            "Spiega fattori che influenzano predizione",
            "Confronta con tassi storici",
            "Fornisce contesto e interpretazione",
            "Indica confidence/affidabilità predizione"
        ]
    }
]


# ===========================================================================
# STATISTICHE DOMANDE
# ===========================================================================

STATISTICS = {
    "total_questions": 10,
    "by_level": {
        "FACILE": 1,
        "FACILE-MEDIO": 2,
        "MEDIO": 2,
        "MEDIO-DIFFICILE": 2,
        "DIFFICILE": 2,
        "MOLTO DIFFICILE": 1
    },
    "by_complexity": {
        "single_tool": 2,  # Q1, Q2
        "2-3_tools": 3,    # Q3, Q4, Q5
        "3-4_tools": 2,    # Q6, Q7
        "5+_tools": 3      # Q8, Q9, Q10
    },
    "skills_tested": [
        "Basic data exploration",
        "Statistical aggregation",
        "Categorical analysis",
        "Missing data handling",
        "Statistical testing",
        "Correlation analysis",
        "ML model training",
        "Feature engineering",
        "Complex filtering",
        "Custom prediction",
        "Result interpretation",
        "Recommendation generation"
    ]
}


# ===========================================================================
# FUNZIONI UTILITY
# ===========================================================================

def print_question(question_dict):
    """Stampa una domanda formattata."""
    print(f"\n{'='*80}")
    print(f"DOMANDA {question_dict['numero']}: {question_dict['livello']}")
    print(f"{'='*80}")
    print(f"\nDomanda:")
    print(f"  {question_dict['domanda']}")
    print(f"\nTool Richiesti:")
    for tool in question_dict['tool_richiesti']:
        print(f"  • {tool}")
    print(f"\nFlusso Atteso:")
    print(f"  {question_dict['flusso_atteso']}")
    print(f"\nComplessità:")
    print(f"  {question_dict['complessità']}")
    print(f"\nOutput Atteso:")
    print(question_dict['output_atteso'])
    print(f"\nCriteri di Successo:")
    for criterio in question_dict['verifica_successo']:
        print(f"  ✓ {criterio}")


def print_all_questions():
    """Stampa tutte le domande."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "10 DOMANDE PROGRESSIVE SUL TITANIC DATASET" + " "*20 + "║")
    print("╚" + "="*78 + "╝")

    for q in TITANIC_QUESTIONS:
        print_question(q)

    # Statistiche finali
    print(f"\n{'='*80}")
    print("STATISTICHE COMPLESSIVE")
    print(f"{'='*80}")
    print(f"\nTotale domande: {STATISTICS['total_questions']}")
    print(f"\nDistribuzione per livello:")
    for level, count in STATISTICS['by_level'].items():
        print(f"  • {level}: {count} domande")
    print(f"\nDistribuzione per complessità:")
    for complexity, count in STATISTICS['by_complexity'].items():
        print(f"  • {complexity}: {count} domande")
    print(f"\nSkills testati:")
    for skill in STATISTICS['skills_tested']:
        print(f"  ✓ {skill}")

    print(f"\n{'='*80}\n")


def get_question(number):
    """Ottieni una domanda specifica."""
    for q in TITANIC_QUESTIONS:
        if q['numero'] == number:
            return q
    return None


def get_questions_by_level(level):
    """Ottieni tutte le domande di un certo livello."""
    return [q for q in TITANIC_QUESTIONS if q['livello'] == level]


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # Mostra una domanda specifica
        num = int(sys.argv[1])
        q = get_question(num)
        if q:
            print_question(q)
        else:
            print(f"Domanda {num} non trovata.")
    else:
        # Mostra tutte le domande
        print_all_questions()

    # Istruzioni
    print("\n" + "─"*80)
    print("Per vedere una domanda specifica: uv run python titanic_questions.py <numero>")
    print("Esempio: uv run python titanic_questions.py 10")
    print("─"*80 + "\n")
