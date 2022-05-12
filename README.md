# Books Success Prediction Experiment

Assuming we were a huge books publisher and a writer came to us with a book, how could we know if this book will be successful?
Also if we were to be the authors of the book, could we ever know if the book will get the audience sympathy or even reach the cinema theatres?
To answer the following questions we set a goal to our research: to see if we can build a model that will predict if a book is so successful that it will also be awarded by list of books features.

## Table of contents

1. <a href="https://chapost1.github.io/books-success-prediction-experiment#introduction">Introduction</a>
2. <a href="https://chapost1.github.io/books-success-prediction-experiment#imports">Imports</a>
3. <a href="https://chapost1.github.io/books-success-prediction-experiment#data_acquisition">Data acquisition</a><br>
    3.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#scraping_challanges">Scraping challanges</a><br>
    3.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#scraping_clean_data">Scraping clean data</a><br>
    3.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#authentication_process">Authentication process</a><br>
    3.4 <a href="https://chapost1.github.io/books-success-prediction-experiment#authentication_class">Authentication class</a><br>
    3.5 <a href="https://chapost1.github.io/books-success-prediction-experiment#scraping_process">Scraping Process</a><br>
    3.6 <a href="https://chapost1.github.io/books-success-prediction-experiment#book_spider_class">Book Spider Class</a><br>
    3.7 <a href="https://chapost1.github.io/books-success-prediction-experiment#scraping_route_creation">Scraping route creation</a><br>
    3.8 <a href="https://chapost1.github.io/books-success-prediction-experiment#genre_spider">Genre spider</a><br>
4. <a href="https://chapost1.github.io/books-success-prediction-experiment#scraping">Scrapping and threading</a><br>
    4.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#first_crawl">First crawl</a><br>
    4.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#concating_data">Concating Data</a><br>
    4.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#data_scraped">Total data scraped</a><br>
5. <a href="https://chapost1.github.io/books-success-prediction-experiment#data_cleaning">Data cleaning</a><br>
    5.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#corrupted_data_cleaning">Corrupted data cleaning</a><br>
    5.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#replace_missing_og">Replace missing data - original title</a><br>
    5.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#none_values">None values - discussion and strategy</a><br>
6. <a href="https://chapost1.github.io/books-success-prediction-experiment#outliers_detection_eda">Pre outliers cleaning EDA</a><br>
    6.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#genre_distribution">Genre distribution</a><br>
    6.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#mean_rating_by_genre">Mean rating by genre</a><br>
    6.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#language_distribution">Language distribution</a><br>
    6.4 <a href="https://chapost1.github.io/books-success-prediction-experiment#edition_count_to_rating">Edition count to rating</a><br>
    6.5 <a href="https://chapost1.github.io/books-success-prediction-experiment#rating_to_award">Rating to award</a><br>
    6.6 <a href="https://chapost1.github.io/books-success-prediction-experiment#pages_count_to_books_count">Pages count to books count</a><br>
7. <a href="https://chapost1.github.io/books-success-prediction-experiment#dealing_with_outliers">Dealing with outliers</a><br>
    7.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#dealing_with_outliers">Outliers detection</a><br>
    7.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#outliers_cleaning">Outliers cleaning</a><br>
    7.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#outliers_clean_results">Outliers cleaning results</a><br>
8. <a href="https://chapost1.github.io/books-success-prediction-experiment#after_clean_eda">EDA after outliers cleaning</a><br>
    8.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#thoughts_after_clean">Thoughts of the results</a><br>
    8.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#metrics_agg">Aggregation metrics</a><br>
    8.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#og_title_eff">Original title correlation with awards</a><br>
    8.4 <a href="https://chapost1.github.io/books-success-prediction-experiment#awards_count_to_genre">Awards count per genre</a><br>
    8.5 <a href="https://chapost1.github.io/books-success-prediction-experiment#awards_book_by_percentage">Awards percentage by genre</a><br>
9. <a href="https://chapost1.github.io/books-success-prediction-experiment#machine_learning_prep">Machine learning preperation</a><br>
10. <a href="https://chapost1.github.io/books-success-prediction-experiment#machine_learning_single">Machine learning - Decision tree</a><br>
    10.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#machine_learning_single">Single decision tree</a><br>
    10.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#first_attempt">First prediction</a><br>
    10.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#ace">New dimenstion - The ace in the sleeve</a><br>
    10.4 <a href="https://chapost1.github.io/books-success-prediction-experiment#depth_opt">Depth optiomazation</a><br>
11. <a href="https://chapost1.github.io/books-success-prediction-experiment#random_forest">Machine learning - Random forest</a><br>
    11.1 <a href="https://chapost1.github.io/books-success-prediction-experiment#overfitting">Overfitting?</a><br>
    11.2 <a href="https://chapost1.github.io/books-success-prediction-experiment#improve_model">Model improvment</a><br>
    11.3 <a href="https://chapost1.github.io/books-success-prediction-experiment#another_ace">Adjusting features</a><br>
    11.4 <a href="https://chapost1.github.io/books-success-prediction-experiment#grid_search_forest">Grid search many forests</a><br>
    11.5 <a href="https://chapost1.github.io/books-success-prediction-experiment#the_top">F-score accuracy addition</a><br>
    11.6 <a href="https://chapost1.github.io/books-success-prediction-experiment#random_states">Random states tests</a><br>
12. <a href="https://chapost1.github.io/books-success-prediction-experiment#conclusion">Conclusion and credits</a><br>    

For implementation, visit hosted notebook:

https://chapost1.github.io/books-success-prediction-experiment/
