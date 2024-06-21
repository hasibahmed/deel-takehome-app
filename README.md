# AI Python Engineer Challenge 


The packages and dependencies are managed by poetry. 

`Caveat:` Pytorch is installed using a wheel specific to the development resource architecture i.e. macosx, python3.11, intel processor (cpu only version is used). Please change to appropriate release for your architecture. In deployment, the app will be containerized and will not have this dependency issue.  
```
> pip install poetry 
> poetry install 
> python app.py
```

## Task 1

The method implemented takes `transaction id` as input and returns the probability [0,1] of a match for each user. To match the string a fuzzy string matching library [RapidFuzz](https://pypi.org/project/rapidfuzz/) is used. Several metrics were investigated. This can potentially be optimized. To improve the performance, a substring corresponding to user name is extracted first from the transaction description. 

Because the method returns a probability, it can be used to pick the best possible match. 


Endpoint: `/match_user`
* Method: GET
* Parameters: `transaction_id` (str)
* Returns: the probability [0,1] of a match for each user in descending order. 
* Limitations: Matching accuracy vary on input data quality, as demonstrated by sub string extraction before comparison. 


## Task 2 
The method implemented takes a `input string` as input and returns the probability [0,1] of matching to each transaction. In this case, semantic similarity is used using language model embedding. Further improvement is made, by picking up the best slice and its cosine similarity with the input string based on a sliding window.

Further improvement is made in execution time, by parallel pool execution (limited by development resources) 

Endpoint: `/similar_transactions`
* Method: GET
* Parameters: `input_str` (str)
* Returns: the probability [0,1] of a match for each user in descending order.
* Limitations: Resource intensive, requires careful resource utilization in the deployment process to reduce execution time. 



### Task 3

PoC to Production: 


* Use a cloud platform that offer scalable computing resources, managed databased and other services supporting deployment, maintenance and monitoring. 

* Use production ready web server e.g. Gunicorn

* Use NGINX or Apache to handle incoming traffic and manage SSL/TLS termination. 

* Use docker to containerize and consistent deployment across various environments. Additionally, Kubernetes can be used to provide auto-scaling and load balancing. 

* Use PostgreSQL for handling large datasets efficiently. 

* Use HTTPS and strong authentication mechanism. 

* Use monitoring tools e.g. Grafana for monitoring performance and health. 