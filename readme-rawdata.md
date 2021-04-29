# Raw Data Folder

## Introduction

> This location stores all of the data that was collected in the form we collected them in. They are a mix of json, csv, and tsv files. There are about 60,000 (or more -- not all were used in this project) of account id's labeled bot/human. There are about 1000 accounts that are labeled "organization"

## Code Samples

> Here is an example of what a json file might look like
```{"created_at": "Mon Apr 16 19:28:33 +0000 2018", "user": {"follow_request_sent": false, "has_extended_profile": false, "profile_use_background_image": false, "default_profile_image": false, "id": 602249341, "profile_background_image_url_https": "https://abs.twimg.com/images/themes/theme4/bg.gif", "verified": false, "translator_type": "none", "profile_text_color": "000000", "profile_image_url_https": "https://pbs.twimg.com/profile_images/923924342974578688/k5RCrlSQ_normal.jpg", "profile_sidebar_fill_color": "000000", "entities": {"url": {"urls": [{"url": "https://t.co/e5t6p9w7D8", "indices": [0, 23], "expanded_url": "http://www.socialresultsltd.com", "display_url": "socialresultsltd.com"}]}, "description": {"urls": []}}, "followers_count": 790, "profile_sidebar_border_color": "000000", "id_str": "602249341", "profile_background_color": "000000", "listed_count": 42, "is_translation_enabled": false, "utc_offset": 3600, "statuses_count": 6252, "description": "Strategic Creative Social Media & Community Engagement Manager. Webeditrix. Digital PR. @sresultsltd #Human #GlutenFree #Short Many opinions\ud83d\udc27\ud83c\udff3\ufe0f\u200d\ud83c\udf08\ud83c\uddea\ud83c\uddfa\ud83c\uddec\ud83c\udde7\ud83c\udff4\ud83e\udd13", "friends_count": 3218, "location": "London - mostly", "profile_link_color": "409FCE", "profile_image_url": "http://pbs.twimg.com/profile_images/923924342974578688/k5RCrlSQ_normal.jpg", "following": false, "geo_enabled": true, "profile_banner_url": "https://pbs.twimg.com/profile_banners/602249341/1517868114", "profile_background_image_url": "http://abs.twimg.com/images/themes/theme4/bg.gif", "screen_name": "EmmaDingle", "lang": "en", "profile_background_tile": false, "favourites_count": 12898, "name": "Emma Dingle\ud83d\udc27\ud83c\udff3\ufe0f\u200d\ud83c\udf08\ud83c\uddea\ud83c\uddfa\ud83c\uddec\ud83c\udde7\ud83c\udff4\ud83e\udd13", "notifications": false, "url": "https://t.co/e5t6p9w7D8", "created_at": "Thu Jun 07 22:16:27 +0000 2012", "contributors_enabled": false, "time_zone": "London", "protected": false, "default_profile": false, "is_translator": false}}```

This json object is a tweet object taken directly from twitter. It represents a single user's tweet.

The tsv files are more simple. Excerpt from ```midterm-2018.tsv```

```
42697610	human
267238769	human
270001939	human
3131769772	human
347504444	human
16358017	human
2423909152	human
327782659	human
254709507	human
412701607	human
3009671708	human
849647464093282305	human
```

The first column represents the id of the twitter account and the second column represents the label assigned to the account.

```org_twitter_handle_2all.csv``` looks the same as the other tsv files but only contains organizations. Note that the first column is the twitter handle for the organization.

```
elpasotxgov,ORGANIZATION
HomeDepot,ORGANIZATION
FannieMae,ORGANIZATION
Chase,ORGANIZATION
cardinalhealth,ORGANIZATION
Merck,ORGANIZATION
statefarm,ORGANIZATION
```

```Organization Twitter Handles.csv``` and ``` Organization-Data_Extra - Sheet 1.csv ``` contain the twitter handles of organizations.

These should be the only files to worry about in this raw dataset. 

Note: This data was processed and put into a master dataset with about ~20k entries that is not in this folder. If you desire to replicate our results, please use the master dataset.