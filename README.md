# Semantic data integration

Team project for the Semantic Data Integration course at the University of Passau, summer semester 2025. Team: Shirin Shoghli, Sanaz Bayat, Shahrzad Torabi and Navid Hadipour Limouei.

We had three spreadsheets about colleges and universities (general data, campuses, national rankings) with different column names and formats. The task was to design one mediated schema for all three and to find which columns match.

What's in here:

- `Datasets/`: the three source files, the mediated schema and the column mapping.
- `Implementation/StringMatchingExample.py`: compares column names with Edit Distance (`thefuzz`), Jaro-Winkler and Word2Vec similarity, and combines the scores with min, max and average.
- `Implementation/ParseXML.py` and the two `Task10_Parsing_XML_*.py` scripts: node-based and path-based matching on the XML versions of the data.
- `Reports/`: our three task reports. Matches are evaluated with precision, recall and F1 there.

## Running it

```bash
pip install numpy pandas thefuzz scipy jarowinkler gensim openpyxl
cd Implementation
python StringMatchingExample.py
python Task10_Parsing_XML_-_Node.py
python Task10_Parsing_XML_-_Path.py
```

The scripts use Windows paths from our machines (`\\`), and the embedding part expects the `glove-twitter-25` vectors from `gensim-data` in your home folder. Adjust the paths before running. `ParseXML.py` reads XML versions of the datasets from `Implementation/XmlFiles/`, which are not in this repository; the two `Task10` scripts have the element paths written into the code and run on their own.
