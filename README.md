# Projet ERO1 : Déneiger Montréal

## Description

Ce projet vise à optimiser les trajets des équipes de déneigement de la ville de Montréal en utilisant des drones pour analyser les niveaux de neige et en planifiant les itinéraires des déneigeuses en fonction des données recueillies. Le projet couvre cinq quartiers de Montréal et utilise des algorithmes avancés pour assurer une couverture efficace et économique des opérations de déneigement.

## Sources

https://donnees.montreal.ca/dataset/geobase

## Installation

1. Installez les dépendances requises :
    ```sh
    pip install -r requirements.txt
    ```

## Utilisation

### Étape 1 : Analyse par Drone

1. Exécutez le script pour simuler le parcours d'un drone dans un quartier en precisant les coordonnees de depart (plusieurs animations seront generees - une pour chaque point de depart):
    ```sh
    Drone/one_postman_length_start_coordinates.py
    ```

2.  Exécutez le script pour simuler le parcours d'un drone dans un quartier en partant de num_start_nodes differents:
    ```sh
    Drone/one_postman_length_diff_start.py
    ```

3. Exécutez le script pour simuler le parcours de deux drones dans un quartier :
    ```sh
    Drone/two_postmen_animation.py
    ```

### Étape 2 : Simulation de Déneigement

1. Exécutez le script pour simuler le parcours d'une déneigeuse respectant les sens de circulation :
    ```sh
    Snowplow/cpp_directed_real_animation.py
    ```

### Étape 3 : Analyse Générale

1. D'autres scripts sont fournis car ils nous ont permis de faire des comparaisons de modelisation ou des verifications 

## Structure des Dossiers

- **Data** : Contient les données brutes.
- **Drone** : Scripts et animations liés à la simulation des drones.
- **General** : Scripts généraux et images pour les analyses et simulations.
- **Miscellanous** : Comparaisons et analyses supplémentaires.
- **Snowplow** : Simulations et animations des déneigeuses.
- **README.md** : Documentation générale du projet.

## Contributeurs

- Baucher Tael
- Chedal-Anglay Berenger
- Godin Nathan
- Hallier Flavien
- Kanounnikoff Martin
