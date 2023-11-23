# Database-Tables

!!! info

    This documeation was generated from the source-code to provide a maximum of consistency.

## clusters

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| uuid | varchar(36) | true | false | 
| project_id | varchar(256) | false | false | 
| owner_id | varchar(256) | false | false | 
| visibility | varchar(10) | false | false | 
| name | varchar(256) | false | false | 

## projects

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| id | varchar(256) | true | false | 
| name | varchar(256) | false | false | 
| creator_id | varchar(256) | false | false | 

## users

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| id | varchar(256) | true | false | 
| name | varchar(256) | false | false | 
| creator_id | varchar(256) | false | false | 
| projects | text | false | false | 
| is_admin | bool | false | false | 
| pw_hash | varchar(64) | false | false | 
| salt | varchar(64) | false | false | 

## dataset

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| uuid | varchar(36) | true | false | 
| project_id | varchar(256) | false | false | 
| owner_id | varchar(256) | false | false | 
| visibility | varchar(10) | false | false | 
| name | varchar(256) | false | false | 
| type | varchar(64) | false | false | 
| location | text | false | false | 

## request_result

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| uuid | varchar(36) | true | false | 
| project_id | varchar(256) | false | false | 
| owner_id | varchar(256) | false | false | 
| visibility | varchar(10) | false | false | 
| name | varchar(256) | false | false | 
| data | text | false | false | 

## checkpoint

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| uuid | varchar(36) | true | false | 
| project_id | varchar(256) | false | false | 
| owner_id | varchar(256) | false | false | 
| visibility | varchar(10) | false | false | 
| name | varchar(256) | false | false | 
| location | text | false | false | 

## error_log

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| timestamp | varchar(128) | false | false | 
| user_id | varchar(256) | false | false | 
| component | varchar(128) | false | false | 
| context | text | false | false | 
| input_values | text | false | false | 
| message | text | false | false | 

## audit_log

| Column-Name | Type | is primary | allow NULL|
| --- | --- | --- | --- |
| timestamp | varchar(128) | false | false | 
| user_id | varchar(256) | false | false | 
| endpoint | varchar(1024) | false | false | 
| request_type | varchar(16) | false | false | 

