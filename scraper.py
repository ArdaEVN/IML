import sys
import csv
from google_play_scraper import Sort, reviews_all

if len(sys.argv) > 1:
    app_id = sys.argv[1]  # Get app ID from the first command-line argument
else:
    app_id = "io.supercent.pizzaidle&hl=en&gl=US"  # Default app ID

all_results = []
for x in range(1, 6):
    results = reviews_all(
        
        app_id,
        sleep_milliseconds=2,  # defaults to 0
        lang='en',  # defaults to 'en'
        country='us',  # defaults to 'us'
        sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
        filter_score_with=x  # defaults to None(means all score)
    )
    all_results.extend(results)

if __name__ == '__main__':
    f = open('{}_reviews.csv'.format(app_id), 'w', encoding="utf8")
    output = csv.writer(f)
    output.writerow(all_results[0].keys())  # header row

    for row in all_results:
        output.writerow(row.values())
    f.close()
