# analizyng-traffic-with-sumo
analyzing traffic with sumo and rl algorithms

Nella cartella nets si trovano i file .net delle reti e i file che descrivono i percorsi .rou  delle varie tipologie di rete. Nella cartella trainingCopiati si trovano gli esperimenti eseguiti copiando la policy dal singolo incrocio, mentre in trainingSIngoli si trovano gli esperimenti in cui ogni semaforo impara da solo. In questi ultimi ci sono script diversi per training e test: nel training si salvano le qtable nella cartella tabelle e in test si leggono queste tabelle. Per eseguire i programmi bisogna posizionarsi da linea di comando nella cartella principale e inserire l'intero percorso dello script a partire dalla cartella principale.
Esempio : python single_intersection/ql_2way-single-intersection.py
