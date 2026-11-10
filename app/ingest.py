from markitdown import MarkItDown
import os

md = MarkItDown()
os.makedirs("data", exist_ok=True)

# Map all majors to their respective Requirements and 4-Year Plan URLs
majors = {
    "Information_Technology": [
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/information-technology/#requirementstext",
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/information-technology/#fouryearplanofstudytext"
    ],
    "Computer_and_Mathematical_Sciences": [
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/computer-and-mathematical-sciences/index.html#requirementstext",
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/computer-and-mathematical-sciences/index.html#fouryearplantext"
    ],
    "Information_Systems": [
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/information-systems/#requirementstext",
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/information-systems/#fouryearplanofstudytext"
    ],
    "Accelerated_BS_MS_Computer_Science": [
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/bsms-computer-science/"
    ],
    "Computer_Science": [
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/computer-science/",
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/computer-information-sciences/computer-science/#fouryearplanofstudytext"
    ],
    "Computer_and_Mathematical_Sciences_Secondary_Education": [
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/mathematicscomputer-and-mathematical-sciences-secondary-education/#requirementstext",
        "https://catalog.towson.edu/undergraduate/fisher-science-mathematics/mathematicscomputer-and-mathematical-sciences-secondary-education/#fouryearplantext"
    ]
}

for major_name, urls in majors.items():
    print(f"Ingesting {major_name}...")
    combined_content = f"# {major_name.replace('_', ' ')} Degree Plan Context\n\n"
    
    for url in urls:
        try:
            result = md.convert(url)
            combined_content += result.text_content + "\n\n---\n\n"
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            
    with open(f"data/{major_name.lower()}.md", "w", encoding="utf-8") as f:
        f.write(combined_content)
        
print("Ingestion complete. Markdown context files are ready for RAG routing.")
