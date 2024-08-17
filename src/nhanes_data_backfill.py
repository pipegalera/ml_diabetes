"""
Downloading the files takes from 15 min to 30 min per year with high speed internet, the good thing is that in can get interrupted at it skips the files already downloaded - just make sure the last file was correctly writen.
"""

import os
from glob import glob
import requests
from bs4 import BeautifulSoup
import io
import pyreadstat
import tempfile
import inquirer
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
PROC_DATA_PATH = os.getenv("PROC_DATA_PATH")

def scrape_nhanes_xpt_files(year, DATA_PATH=RAW_DATA_PATH):
    """
    Note PAXMIN.XPT aka "Physical Activity Monitor - Minute	" is the only file missing.
    It's +6 gigas and CDC website its not preciselly fast.
    It takes 6 hours to download usually.
    Polling data without a unique identifer also will be missing ("*POL*.parquet") since
    I have no use for it.
    """

    list_types = ["Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire"]

    # Create folder structure for the data based on the year
    os.chdir(DATA_PATH)
    os.makedirs(f"{year}", exist_ok=True)
    os.chdir(f"{year}")

    print(f"NHANES Data from {year} year")
    print("__________________________")
    print("__________________________")

    for type in list_types:
        # Type of data and year
        url = f"https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={type}&CycleBeginYear={year[:4]}"

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            continue

        # Create folder structure for the data based on the data type
        os.makedirs(type, exist_ok=True)
        os.chdir(type)
        print("### Data type:", type, "###")
        print("__________________________")

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all links that end with .XPT (case insensitive)
        xpt_links = soup.find_all('a', href=lambda href: href and href.lower().endswith('.xpt'))

        # Download and process each XPT file
        for link in xpt_links:
            file_url = link['href']
            if not file_url.startswith('http') and not "PAXMIN" in file_url and not "POL" in file_url:
                file_url = f"https://wwwn.cdc.gov{file_url}"
            else:
                continue

            file_name = file_url.split('/')[-1]
            parquet_filename = file_name.replace('.XPT', '.parquet')

            # Download the XPT file if it doesn't exist
            if not os.path.exists(parquet_filename):
                print(f"Downloading {file_name} from CDC website...")
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    # Create a temporary file so pyreadstat can read it
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.XPT') as temp_file:
                        temp_file.write(file_response.content)
                        temp_file_path = temp_file.name

                    xpt_data, _ = pyreadstat.read_xport(temp_file_path, encoding='cp1252')
                    xpt_data.to_parquet(parquet_filename, index=False)

                    # Remove the temporary file
                    os.unlink(temp_file_path)
                    print(f"Saved as {parquet_filename}")
                else:
                    print(f"Failed to download {file_name}. Status code: {file_response.status_code}")
            else:
                print(f"{parquet_filename} file already in the destination folder")

        # Moving again to parent directory
        os.chdir('..')
        print("__________________________")


def main():
    years = ["1999-2000", "2001-2002", "2003-2004",
        "2005-2006", "2007-2008", "2009-2010", "2011-2012","2013-2014"]

    questions = [
        inquirer.Checkbox(
            'selected_years',
            message="Select the years to scrape from NHANES website (takes from 15 min to 30 min per year.)",
            choices=years,
            default=years
        )]

    answers = inquirer.prompt(questions)
    selected_years = answers['selected_years']

    for year in selected_years:
        scrape_nhanes_xpt_files(year)

if __name__ == "__main__":
    main()
