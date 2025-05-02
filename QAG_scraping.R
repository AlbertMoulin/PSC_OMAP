library(rvest)
library(httr)
library(xml2)

# Function to scrape data from the page
scrape_data <- function(url_part) {
  # Construct the full URL
  base_url <- "https://www.assemblee-nationale.fr/dyn/17/questions/QANR5L17QE"
  full_url <- paste0(base_url, url_part)
  
  # Send a GET request to the webpage
  webpage <- read_html(full_url)
  
  # Extract the information you need (Example: all <h1> and <p> tags)
  date_question <- webpage %>%
    html_nodes("._regular:nth-child(2) .link") %>%
    html_text()
  
  question_text <- webpage %>%
    html_nodes("._colored-primary+ ._pa-small") %>%
    html_text()

  date_answer <- webpage %>%
    html_nodes("._regular~ ._regular .link") %>%
    html_text()

  answer_text <- webpage %>%
    html_nodes("._colored-ultralightgrey ._pa-small") %>%
    html_text()

  

if (length(date_question) > length(date_answer)) {
  date_answer <- c(date_answer, rep(NA))
}

if (length(question_text) > length(answer_text)) {
  answer_text <- c(answer_text, rep(NA))
}



# Créer le data.frame
df <- data.frame()



  result_df <- data.frame(
    date_question = date_question,
    question_text = question_text,
    date_answer = date_answer,
    answer_text = answer_text,
    stringsAsFactors = FALSE
  )
  
  
  return(result_df)
}
print("hello")
# Initialize an empty list to store results
all_results <- list()
nb_pages <- 6178 #à modifier pour avoir toutes les questions

# For loop to call the function and grow the list
for (i in 1:nb_pages) {
  # Call the scrape_data function
  url_part <- as.character(i)
  scraped_data <- scrape_data(url_part)
  
  # Append the result to the all_results list
  all_results <- append(all_results, list(scraped_data))
  if (i %% 100 == 0) {
    cat("Scraped page", i, "\n")
  }
}

final_data <- do.call(rbind, all_results)

# Save the final data frame as a CSV file on your computer
write.csv(final_data, "donnees_fusionnees_XVII.csv", row.names = FALSE)

# Print a message indicating the file was saved
cat("Data has been saved")

