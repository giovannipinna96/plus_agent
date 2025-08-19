# SLURM Test Scripts

Questa cartella contiene script SLURM per testare sistematicamente tutti i componenti del sistema multi-agent. Gli script sono ordinati per complessità crescente e dipendenze.

## Prerequisiti

Prima di eseguire gli script, assicurati di avere:

1. **Ambiente Python configurato:**
   ```bash
   module load Python/3.9-GCCcore-11.2.0
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Dataset Titanic scaricato:**
   ```bash
   cd plus_agent
   python data/download_titanic.py
   ```

3. **Directory logs creata:**
   ```bash
   mkdir -p logs
   ```

4. **Variabili d'ambiente configurate** (opzionale, gli script hanno valori di default):
   ```bash
   export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
   export DEVICE="auto"
   export LANGSMITH_TRACING="false"  # per i test
   ```

## Script Disponibili

### 1. test_llm_wrapper.slurm
**Scopo:** Testa il wrapper LLM e l'inizializzazione dei modelli

**Risorse:**
- Partition: `gpu`
- Memoria: `16G`
- GPU: `1`
- Tempo: `30 minuti`

**Cosa testa:**
- Caricamento del modello Hugging Face
- Inizializzazione del wrapper LangChain
- Generazione di testo di base

**Come lanciare:**
```bash
sbatch bash/test_llm_wrapper.slurm
```

### 2. test_data_tools.slurm  
**Scopo:** Testa tutti i tool per la lettura e analisi dei dati

**Risorse:**
- Partition: `cpu`
- Memoria: `8G`
- Tempo: `15 minuti`

**Cosa testa:**
- Lettura file CSV
- Analisi colonne e tipi di dati
- Generazione sommari statistici
- Anteprima dati

**Come lanciare:**
```bash
sbatch bash/test_data_tools.slurm
```

### 3. test_manipulation_tools.slurm
**Scopo:** Testa i tool per la manipolazione e preprocessing dei dati

**Risorse:**
- Partition: `cpu`
- Memoria: `8G`
- Tempo: `15 minuti`

**Cosa testa:**
- Gestione valori mancanti
- Creazione variabili dummy
- Conversione tipi di dati
- Transformazioni dei dati

**Come lanciare:**
```bash
sbatch bash/test_manipulation_tools.slurm
```

### 4. test_operations_tools.slurm
**Scopo:** Testa i tool per operazioni matematiche e analisi

**Risorse:**
- Partition: `cpu`
- Memoria: `8G`
- Tempo: `15 minuti`

**Cosa testa:**
- Filtraggi dei dati
- Operazioni matematiche (media, somma, ecc.)
- Aggregazioni per gruppo
- Operazioni complesse

**Come lanciare:**
```bash
sbatch bash/test_operations_tools.slurm
```

### 5. test_ml_tools.slurm
**Scopo:** Testa i tool per machine learning

**Risorse:**
- Partition: `gpu`
- Memoria: `16G`
- GPU: `1`
- Tempo: `30 minuti`

**Cosa testa:**
- Training Random Forest
- Training SVM
- Training K-NN
- Valutazione modelli
- Salvataggio/caricamento modelli

**Come lanciare:**
```bash
sbatch bash/test_ml_tools.slurm
```

### 6. test_single_agents.slurm
**Scopo:** Testa ogni agente individualmente

**Risorse:**
- Partition: `gpu`
- Memoria: `32G`
- GPU: `1`
- Tempo: `1 ora`

**Cosa testa:**
- PlannerAgent: creazione piani di esecuzione
- DataReaderAgent: analisi dati
- DataManipulationAgent: preprocessing
- DataOperationsAgent: operazioni matematiche
- MLPredictionAgent: training modelli

**Come lanciare:**
```bash
sbatch bash/test_single_agents.slurm
```

### 7. test_orchestrator.slurm
**Scopo:** Testa l'orchestratore multi-agent con LangGraph

**Risorse:**
- Partition: `gpu`
- Memoria: `32G`
- GPU: `1`
- Tempo: `1.5 ore`

**Cosa testa:**
- Workflow semplici (singolo agente)
- Workflow complessi (multi-agente)
- Gestione stato tra agenti
- Routing supervisor
- Gestione errori

**Come lanciare:**
```bash
sbatch bash/test_orchestrator.slurm
```

### 8. test_gradio_interface.slurm
**Scopo:** Testa l'interfaccia Gradio senza avviarla

**Risorse:**
- Partition: `gpu`
- Memoria: `32G`
- GPU: `1`
- Tempo: `2 ore`

**Cosa testa:**
- Inizializzazione sistema
- Gestione upload file
- Processamento richieste utente
- Integrazione con orchestratore
- Funzioni helper dell'interfaccia

**Come lanciare:**
```bash
sbatch bash/test_gradio_interface.slurm
```

### 9. test_full_system.slurm
**Scopo:** Test end-to-end dell'intero sistema

**Risorse:**
- Partition: `gpu`  
- Memoria: `64G`
- GPU: `1`
- Tempo: `3 ore`

**Cosa testa:**
- Test prompts di tutte le complessità
- Workflow completi end-to-end
- Performance del sistema
- Integrazione di tutti i componenti
- Test con prompt reali dal file test_prompts.py

**Come lanciare:**
```bash
sbatch bash/test_full_system.slurm
```

## Ordine di Esecuzione Raccomandato

Per testare sistematicamente il sistema, esegui gli script in questo ordine:

1. **Prima i componenti base:**
   ```bash
   sbatch bash/test_llm_wrapper.slurm
   sbatch bash/test_data_tools.slurm
   ```

2. **Poi i tool specializzati:**
   ```bash
   sbatch bash/test_manipulation_tools.slurm
   sbatch bash/test_operations_tools.slurm  
   sbatch bash/test_ml_tools.slurm
   ```

3. **Quindi gli agenti:**
   ```bash
   sbatch bash/test_single_agents.slurm
   ```

4. **Infine l'integrazione completa:**
   ```bash
   sbatch bash/test_orchestrator.slurm
   sbatch bash/test_gradio_interface.slurm
   sbatch bash/test_full_system.slurm
   ```

## Monitoring dei Job

### Controllare stato dei job:
```bash
squeue -u $USER
```

### Vedere output in tempo reale:
```bash
tail -f logs/test_full_system_[JOB_ID].out
```

### Cancellare un job:
```bash
scancel [JOB_ID]
```

## Interpretazione dei Risultati

### ✅ Test Passati
- Exit code 0
- Log contiene "completed successfully"
- Nessun errore Python nei log

### ❌ Test Falliti  
- Exit code non-zero
- Errori Python o SLURM nei log
- Timeout del job

### ⚠️ Test Parziali
- Alcuni componenti funzionano, altri no
- Warning ma non errori fatali
- Performance sotto le aspettative

## Debugging

### Log Files
- `logs/test_*_[JOB_ID].out`: Output standard
- `logs/test_*_[JOB_ID].err`: Errori SLURM/sistema

### Common Issues
1. **OOM (Out of Memory)**: Aumenta `--mem` nello script
2. **GPU non disponibile**: Controlla con `sinfo -p gpu`
3. **Modello non scaricato**: Controlla cache HuggingFace
4. **Import errors**: Verifica PYTHONPATH e installazione dipendenze

### Modalità Debug
Per debug dettagliato, modifica gli script aggiungendo:
```bash
export PYTHONPATH=".:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1  # per debug GPU
python -u [script.py]  # unbuffered output
```

## Performance Benchmarks

### Tempi Attesi (su V100):
- LLM wrapper: ~2-5 minuti
- Data tools: ~1-2 minuti  
- ML tools: ~5-10 minuti
- Single agents: ~15-30 minuti
- Full system: ~60-120 minuti

### Memoria Tipica:
- Tool tests: ~2-8 GB
- Agent tests: ~8-16 GB
- Full system: ~16-32 GB