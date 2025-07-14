import clayrs.content_analyzer as ca
import clayrs.evaluation as eva
import pandas as pd
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def eval_recommendations(dataset='amazon_elec', ks=[5, 10], relevant_threshold=1):
    """
    Valuta le raccomandazioni generate dai modelli LightGCN e BPR
    """
    # Cartelle delle predizioni
    preds_folders = {
        #'top5': 'code_recbole/preds/top5',
        #'top10': 'code_recbole/preds/top10'
        'top5': 're-ranking/reranked_embedding_miniLM/top5',
        'top10': 're-ranking/reranked_embedding_miniLM/top10',
        'top1': 're-ranking/reranked_embedding_miniLM/full',
    }
    
    # Modelli da valutare
    models = ['LightGCN-Jun-11-2025_11-23-41.pth', 'LightGCN-Jun-11-2025_11-38-53.pth', 'LightGCN-Jun-11-2025_12-09-16.pth', 'BPR-Jun-11-2025_11-18-24.pth']
    
    # Carica i dati di training e test
    print(f"Caricamento dati per {dataset}...")
    train_data = ca.CSVFile(os.path.join('datasets', dataset, "train.tsv"), separator="\t")
    train_ratings = ca.Ratings(train_data)
    
    test_data = ca.CSVFile(os.path.join('datasets', dataset, "test.tsv"), separator="\t")
    test_ratings = ca.Ratings(test_data)
    
    # Dizionario per salvare tutti i risultati
    all_results = {}
    
    # Itera sui diversi k (top-5, top-10)
    for k in ks:
        folder_key = f'top{k}'
        preds_folder = preds_folders[folder_key]
        
        print(f"\n=== Valutazione per K={k} ===")
        
        # Controlla quali file di predizione esistono
        if not os.path.exists(preds_folder):
            print(f"Cartella {preds_folder} non trovata!")
            continue
            
        prediction_files = []
        for model in models:
            pred_file = f"{model}.tsv"
            if os.path.exists(os.path.join(preds_folder, pred_file)):
                prediction_files.append(pred_file)
            else:
                print(f"File {pred_file} non trovato in {preds_folder}")
        
        print(f'File di predizione trovati: {prediction_files}')
        
        # Definisci le metriche da calcolare
        metric_list = [
            eva.PrecisionAtK(k=k, relevant_threshold=relevant_threshold),
            eva.RecallAtK(k=k, relevant_threshold=relevant_threshold),
            eva.FMeasureAtK(k=k, relevant_threshold=relevant_threshold),
            eva.NDCGAtK(k=k),
            eva.GiniIndex(),
            #eva.EPC(k=k, original_ratings=train_ratings, ground_truth=test_ratings),
            #eva.APLT(k=k, original_ratings=train_ratings),
        ]
        
        # Dizionario per i risultati di questo k
        k_results = {}
        
        # Valuta ogni file di predizione
        for pred_file in tqdm(prediction_files, desc=f"Valutazione Top-{k}"):
            try:
                # Carica il file delle predizioni
                pred_path = os.path.join(preds_folder, pred_file)
                eval_summary = ca.CSVFile(pred_path, separator="\t")
                
                # Prepara le liste per la valutazione
                truth_list = [test_ratings]
                rank_list = [ca.Rank(eval_summary)]
                
                # Crea il modello di valutazione
                em = eva.EvalModel(
                    pred_list=rank_list,
                    truth_list=truth_list,
                    metric_list=metric_list
                )
                
                # Calcola le metriche
                sys_result, users_result = em.fit()
                sys_result = sys_result.loc[['sys - mean']]
                sys_result.reset_index(drop=True, inplace=True)
                
                # Aggiungi informazioni sul modello
                model_name = pred_file.replace(f'_top{k}.tsv', '')
                sys_result['model'] = model_name
                sys_result['k'] = k
                
                # Pulisci i nomi delle colonne
                sys_result.columns = [x.replace(" - macro", "") for x in sys_result.columns]
                
                # Riordina le colonne
                cols = list(sys_result.columns)
                cols = cols[-2:] + cols[:-2]  # Metti 'model' e 'k' all'inizio
                sys_result = sys_result.loc[:, cols]
                
                k_results[model_name] = sys_result
                print(f"✓ {model_name} valutato con successo")
                
            except Exception as e:
                print(f"✗ Errore nella valutazione di {pred_file}: {str(e)}")
                continue
        
        # Salva i risultati per questo k
        if k_results:
            k_df = pd.concat([v for v in k_results.values()]).reset_index(drop=True)
            k_df = k_df.sort_values(by=['model'], ascending=[True])
            
            # Salva i risultati
            os.makedirs('results_embedding_diversify', exist_ok=True)
            output_file = f'results_embedding_diversify/{dataset}_top{k}_results.tsv'
            k_df.to_csv(output_file, index=False, sep='\t')
            print(f"Risultati salvati in: {output_file}")
            
            # Mostra un summary dei risultati
            print(f"\nRisultati Top-{k}:")
            #print(k_df[['model', 'Precision@5', 'Recall@5']].round(4))
            all_results[f'top{k}'] = k_df
    
    # Combina tutti i risultati
    if all_results:
        combined_results = pd.concat([v for v in all_results.values()]).reset_index(drop=True)
        combined_output = f'results_embedding_diversify/{dataset}_all_results.tsv'
        combined_results.to_csv(combined_output, index=False, sep='\t')
        print(f"\nTutti i risultati salvati in: {combined_output}")
        
        return combined_results
    else:
        print("Nessun risultato generato!")
        return None

def prepare_dataset_for_clayrs(dataset_name='amazon_elec'):
    """
    Converte i file .inter in formato TSV per ClayRS
    """
    print(f"Preparazione dataset {dataset_name} per ClayRS...")
    
    # Crea la cartella datasets se non esiste
    dataset_path = f'datasets/{dataset_name}'
    os.makedirs(dataset_path, exist_ok=True)
    
    # Percorsi dei file RecBole
    base_path = f'code_recbole/dataset/{dataset_name}'
    
    # Converti i file
    files_to_convert = {
        'train': f'{base_path}/{dataset_name}.train.inter',
        'test': f'{base_path}/{dataset_name}.test.inter'
    }
    
    for split, file_path in files_to_convert.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep='\t')
            
            # Assicurati che le colonne abbiano i nomi corretti per ClayRS
            # ClayRS si aspetta: user_id, item_id, rating (o score)
            if 'user_id:token' in df.columns:
                df = df.rename(columns={'user_id:token': 'user_id'})
            if 'item_id:token' in df.columns:
                df = df.rename(columns={'item_id:token': 'item_id'})
            if 'rating:float' in df.columns:
                df = df.rename(columns={'rating:float': 'rating'})
            
            output_path = f'{dataset_path}/{split}.tsv'
            df.to_csv(output_path, sep='\t', index=False)
            print(f"✓ {split}.tsv creato ({len(df)} righe)")
        else:
            print(f"✗ File {file_path} non trovato!")

if __name__ == "__main__":
    # Prepara il dataset per ClayRS
    prepare_dataset_for_clayrs('amazon_elec')
    
    # Esegui la valutazione
    print("\n" + "="*50)
    print("INIZIO VALUTAZIONE")
    print("="*50)
    
    results = eval_recommendations(
        dataset='amazon_elec',
        ks=[1, 5, 10],
        relevant_threshold=1
    )
    
    if results is not None:
        print("\n" + "="*50)
        print("VALUTAZIONE COMPLETATA")
        print("="*50)
        print(f"Risultati finali:\n{results}")
    else:
        print("\n" + "="*50)
        print("VALUTAZIONE FALLITA")
        print("="*50)