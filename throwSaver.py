"""
Module for saving real world throw data into a .csv file to be read by other modules.
Currently configured players are me and my roommates. It is possible to configure new
players, but it would need extensive rewrites. 
"""
import pandas as pd
import sys
import os

CWD = os.getcwd()
SAVENAME = "dfThrowing.csv"
SAVEPATH = f"{CWD}/{SAVENAME}"
LOADPATH = SAVEPATH


# the actual parser for throws
def throwParser() -> dict:

    print("type in number and an initial in the form of **throwValue*, *initial** to update the dictionary")
    data = {"A": [], "M": [], "T": [], "K": []}
    counter = 0
    while True:
        inputed = input(f"enter data for throw {counter}:\n")
        inputed.strip()
        print(f"current input = {inputed}")
        if inputed.upper() == "END":
            print("Exiting loop...")
            break
        else:
            inputed = inputed.split(",")
            inputed = [i.strip().upper() for i in inputed]
            print(inputed)
            try:
                data[inputed[1]].append(int(inputed[0]))
            except (KeyError, ValueError, IndexError):
                print("Please add a valit number and initial pair, or 'end' statement to exit the parser loop")
                continue
            counter += 1
        print(data)

    return data


def dataFramer(dictionary: dict) -> pd.DataFrame:

    data = dictionary
    # wait, why am I doing this shit, as in why am I changing the data type of value
    df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data.items()]))

    return df


def saveData(dataFrame: pd.DataFrame, includeHeader: bool):
    dataFrame.to_csv(SAVEPATH, mode="a", index=False, na_rep="NaN", header=includeHeader)


def retrieveData() -> pd.DataFrame:
    return pd.read_csv(LOADPATH).fillna(0).astype(int)


def main():
    while True:
        inputOption = input("Would you like to r(E)trieve old data, or are you (P)laying again?\n").strip().upper()
        if inputOption == "E":
            try:
                (print(retrieveData()))
            except (IOError, FileNotFoundError):
                print("Error in data retrieval, .csv file is either missing or inaccessible")
            sys.exit()

        elif inputOption == "P":
            try:
                results = throwParser()
                data = dataFramer(results)
                print(f"The resultant data frame:\n{data}\n")

                saveOption = input("Would you like to save the data Y/N?\n")
                if saveOption.strip().upper() == "Y":

                    header = input("Would you like to include the header Y/N?\n").strip().upper()
                    try:
                        if header == "Y":
                            saveData(data, True)
                            print(f"Data with header has been saved to: {SAVEPATH}\n")
                        elif header == "N":
                            saveData(data, False)
                            print(f"Just the throw values were saved to: {SAVEPATH}\n")
                    except(KeyError, ValueError, IndexError):
                        print("please input either Y or N")

            except(KeyError, ValueError, IndexError) as e:
                print(f"Error: {str(e)}. Please try again")
            except(Exception) as e:
                print(f"An unexpected error occured: {str(e)}.")

            sys.exit()


while __name__ == "__main__":
    main()

