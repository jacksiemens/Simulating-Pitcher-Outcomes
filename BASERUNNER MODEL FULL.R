
# SIMULATING PITCHER SEASONS 

library(baseballr)
library(purrr)
library(lubridate)
library(furrr)
library(foreach)
library(doParallel)
library(tidyverse)
library(randomForest)
library(doFuture)




# This section is not the most effective solution, but runs quick enough 

# This creates a data frame of daily periods, where there is a row for every
# day between the start date and end date.
# When importing pitch data, we need to split the data to make sure we don't
# overwhelm the api. 

create_date_ranges <- function(start_date, end_date){
  start_date <- ymd(start_date)
  end_date <- ymd(end_date)
  total_days <- as.integer(end_date - start_date)
  
  date_ranges <- tibble(
    start = as.Date(character(total_days)),
    end = as.Date(character(total_days))
  )
  for (i in 1:total_days) {
    if (i == total_days) {
      date_ranges$start[i] <- start_date + (i-1)
      date_ranges$end[i] <- end_date
    } else {
      date_ranges$start[i] <- start_date + (i-1)
      date_ranges$end[i] <- start_date + i - 1
    }
  }
  return(date_ranges)
}

dates_2021 <- create_date_ranges("2021-04-01",
                                  "2021-10-05")
dates_2022 <- create_date_ranges("2022-04-07",
                                 "2022-10-02")
date_ranges <- bind_rows(dates_2021 , dates_2022 )

# Use date ranges to create list of pitches, 
# Spread work across multiple cores

pitches_date_ranges <- function(date_ranges){
  pitches <- tibble()
  
  # Set up parallel backend with multiple cores
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  
  # Load the baseballr package on the parallel workers
  clusterEvalQ(cl, library(baseballr))
  
  # Define function to retrieve statcast data for a given date range
  get_pitches <- function(start, end) {
    statcast_search(start_date = start, end_date = end)
  }
  
  # Export the functions to the parallel workers
  clusterExport(cl, c("get_pitches", "statcast_search"))
  
  # Use foreach to execute the loop in parallel
  pitches <- foreach(i = 1:nrow(date_ranges)) %dopar% {
    range_pitches <- get_pitches(date_ranges$start[i], date_ranges$end[i])
  }
  
  # Stop the parallel backend
  stopCluster(cl)
  
  # Combine the results into a single dataframe
  pitches <- do.call(bind_rows, pitches)
  
  
  pitches <- discard(pitches , ~ nrow(.x) == 0)
  
  return(pitches)
  
}



# Create ball_in_play, and HR models 
# ball_in_play will predict singles, XBH

# Explanation of relevant variables

# game_pk - the unique ID number for the game
# at_bat_number - the ID for the plate appearance. Can be combined with game_pk for a complete unique id. 
# stand - whether a hitter is left/right handed (indicates shifting location of fielders)
# launch_speed - Exit velocity of the batted ball in mph
# launch_angle - Launch Angle of batted ball in degrees. 
# hc_x - the x-coordinate of the landing position of the batted ball
# hc_y - the y-coordinate of the landing position of the batted ball (invert for visualization)
# outs_when_up - number of outs recorded prior to plate appearance
# of_fielding_allignment - Indicates the configuration of the defensive outfield allignment
# if_fielding_allignment - Indicates the configuration of the defensive infield allignment
# events - result of plate appearance

# Venue, Defensive Ability of opposition, and various weather factors 
# are excluded despite being predictive
# (the goal of this project is to assess pitching, not environmental factors) 

df <-  bind_rows(pitches)%>%
  select(game_pk, at_bat_number, stand, launch_speed,launch_angle, hc_x, hc_y, outs_when_up, of_fielding_alignment,
         if_fielding_alignment, events) %>%
  filter(events != "") %>%
  mutate(home_run = ifelse(events == "home_run", 1, 0),
         events = ifelse(events == "single", "single",
                         ifelse(events %in% c("double", "triple"), "xbh", "x" ))
  ) %>%
  na.omit()

df$events <- as.factor(df$events)
df$stand <- as.factor(df$stand)
df$of_fielding_alignment <- as.factor(df$of_fielding_alignment)
df$if_fielding_alignment <- as.factor(df$if_fielding_alignment)


# Define Training / Testing Data Set
# Size of set was based on previous experience/ knowledge

set.seed(403)
train_index <- sample(1:nrow(df), 0.075 * nrow(df))
train_set <- df[train_index, ]
test_set <- df[-train_index, ]

bip_model <- randomForest(events ~ launch_speed + launch_angle +  hc_x + hc_y+
                            stand + of_fielding_alignment + if_fielding_alignment,
                          data = train_set)
# 97.9 MB for reference
# OOB estimate of  error rate: 10.81%
#Confusion matrix:
#        single     x  xbh class.error
#single   4642  1416  159  0.25333762
#x         484 21064  134  0.02850291
#xbh       323   727 1051  0.49976202

hr_model <- randomForest(home_run ~ launch_speed + launch_angle +  hc_x + hc_y,
                         data =train_set)
# 67.2 MB for reference
# Mean of squared residuals: 0.009972438
# % Var explained: 77.3



# Creating model to simulate base running 

# For example, there is a runner on first, the ball is hit 
# to the the corner, where will the runner end up??

# What do we need?
# outs, PA result. If in play, we need the hit data.
# ex. launch speed, launch angle, spray angle
# Base running data -  runner on first, runner on second,
# runner on third. infield allignment.
# batter handedness. 
# And then the result of the baserunner.
# Where did they end up? Did they score? What is the final end state.



pitches_df <- bind_rows(pitches) %>%
  filter(game_type == "R")%>%
  arrange(game_date, game_pk, at_bat_number, desc(pitch_number)) %>%
  select(game_date, game_pk, inning_topbot, inning, at_bat_number, player_name, 
         batter, events, description, 
         des, stand, bb_type, on_1b, on_2b, on_3b, outs_when_up,
         bat_score, post_bat_score, hc_x, hc_y, launch_speed, launch_angle,
         if_fielding_alignment, of_fielding_alignment) %>%
  filter(events != "") %>%
  rename(start_on_1b = on_1b, start_on_2b = on_2b, start_on_3b = on_3b ) %>%
  mutate(runs_scored = post_bat_score - bat_score,
         start_on_1b = ifelse(is.na(start_on_1b), FALSE, TRUE),
         start_on_2b = ifelse(is.na(start_on_2b), FALSE, TRUE),
         start_on_3b = ifelse(is.na(start_on_3b), FALSE, TRUE))

pitches_df$end_on_1b <- ""
pitches_df$end_on_2b <- ""
pitches_df$end_on_3b <- ""
pitches_df$outs_created <- ""
pitches_df$bb_type[pitches_df$bb_type == ""] <- NA


# Creating the Unique Innings Data Set
pb <- progress_bar$new(total = length(unique(pitches_df$game_pk)), format = "[:bar] :percent :eta")
unique_innings <- list()
# Creating the Unique Innings 
for (game_pk in unique(pitches_df$game_pk)){
  #load game data
  game <- pitches_df[pitches_df$game_pk == game_pk,]
  
  # now loop through the innings 
  
  for (inning in unique(game$inning)){
    inning_data <- game[game$inning == inning, ]
    
    
    # now loop through each top/ bottom
    
    for (half_inning in c("Top", "Bot")){
      
      half_inning_data <- inning_data[inning_data$inning_topbot == half_inning, ]
      
      
      unique_innings[[paste(game_pk, half_inning, inning, sep = "-")]] <- half_inning_data
    }
    
  }
  pb$tick()
}


# Completing the End State Data for each PA
pb <- progress_bar$new(total = length(unique_innings), format = "[:bar] :percent :eta")

for (inning in 1:length(unique_innings)){
  
  for (i in 1:nrow(unique_innings[[names(unique_innings)[inning]]])){
    if((i + 1) > nrow (unique_innings[[names(unique_innings)[inning]]])){
      
      unique_innings[[names(unique_innings)[inning]]]$end_on_1b[i] <- unique_innings[[names(unique_innings)[inning]]]$start_on_1b[i]
      unique_innings[[names(unique_innings)[inning]]]$end_on_2b[i] <- unique_innings[[names(unique_innings)[inning]]]$start_on_2b[i]
      unique_innings[[names(unique_innings)[inning]]]$end_on_3b[i] <- unique_innings[[names(unique_innings)[inning]]]$start_on_3b[i]
      unique_innings[[names(unique_innings)[inning]]]$outs_created[i] <- 3 - unique_innings[[names(unique_innings)[inning]]]$outs_when_up[i] 
    }
    else{
      unique_innings[[names(unique_innings)[inning]]]$end_on_1b[i] <- unique_innings[[names(unique_innings)[inning]]]$start_on_1b[i + 1]
      unique_innings[[names(unique_innings)[inning]]]$end_on_2b[i] <- unique_innings[[names(unique_innings)[inning]]]$start_on_2b[i + 1]
      unique_innings[[names(unique_innings)[inning]]]$end_on_3b[i] <- unique_innings[[names(unique_innings)[inning]]]$start_on_3b[i + 1]
      unique_innings[[names(unique_innings)[inning]]]$outs_created[i] <- unique_innings[[names(unique_innings)[inning]]]$outs_when_up[i + 1] - unique_innings[[names(unique_innings)[inning]]]$outs_when_up[i] 
    }
  }
  pb$tick()
}



batted_balls <- bind_rows(unique_innings) %>%
  select(stand, start_on_1b,start_on_2b, start_on_3b, outs_when_up, outs_when_up,
         hc_x, hc_y, launch_speed, launch_angle, if_fielding_alignment, of_fielding_alignment, 
         end_on_1b, end_on_2b, end_on_3b, outs_created, runs_scored ) 
 
batted_balls$end_on_1b <- as.logical(batted_balls$end_on_1b)
batted_balls$end_on_2b <- as.logical(batted_balls$end_on_2b)
batted_balls$end_on_3b <- as.logical(batted_balls$end_on_3b)
batted_balls$outs_created <- as.integer(batted_balls$outs_created )
batted_balls$stand <- as.factor(batted_balls$stand)
batted_balls$of_fielding_alignment <- as.factor(batted_balls$of_fielding_alignment)
batted_balls$if_fielding_alignment <- as.factor(batted_balls$if_fielding_alignment)

batted_balls <- na.omit(batted_balls)


# This decision was the toughest, most time consuming of the 
# whole process. This was the eventual solution to dealing with
# the dependant probabilites of various outcomes. 
# It's much more of a black box method that previous attempts that
# strategies the various outcomes independently

batted_balls<- batted_balls %>%
  mutate(end_on_1b  = as.numeric(end_on_1b),
         end_on_2b  = as.numeric(end_on_2b),
         end_on_3b  = as.numeric(end_on_3b)) %>%
  mutate(end_state = paste(as.character(end_on_1b),
                           as.character(end_on_2b),
                           as.character(end_on_3b), 
                           as.character(outs_created),
                           as.character(runs_scored),
                           sep = "_"))%>%
  select(-end_on_1b, -end_on_2b, -end_on_3b)



batted_balls$end_state <- as.factor(batted_balls$end_state)
# This creates an end_state that would read like 
# 1b_2b_3b_outs_runs. A grand slam would read as
# 0_0_0_0_4, while an infield popup with the bases loaded would be
# 1_1_1_1_0

# Making Predictions
bip_pred <- predict( bip_model, batted_balls, type = "prob")
hr_pred <- predict(hr_model, batted_balls)

batted_balls$xHR <- round(as.numeric(hr_pred), 8)
batted_balls <- cbind(batted_balls,bip_pred ) %>%
  select(stand:of_fielding_alignment, xHR,single, xbh, outs_created:end_state) %>%
  rename(xSingle =single, xXBH =  xbh)


# Now lets train the final, end state predictor model
set.seed(123)
es_train_index <- sample(1:nrow(batted_balls), 0.8 * nrow(batted_balls))
es_train_set <- batted_balls[es_train_index, ]
es_test_set <- batted_balls[-es_train_index, ]

# need to refactor to remove missing factor types
#(rare end states that don't occur in the training)

es_train_set $end_state <- factor(es_train_set$end_state)

end_state_model <- randomForest(end_state ~ stand+ start_on_1b+start_on_2b+ start_on_3b+ outs_when_up+ 
                                  hc_x+ hc_y+ launch_speed+ launch_angle+ if_fielding_alignment+ of_fielding_alignment + xHR + xSingle+ xXBH,
                                data = es_train_set, proximity = FALSE)
# Model is 651 MB
# OOB estimate of  error rate: 20%
# Importance:
#                           MeanDecreaseGini
# stand                         696.9674
# start_on_1b                 17489.9157
# start_on_2b                 11256.2776
# start_on_3b                  6133.0998
# outs_when_up                 3713.1373
# hc_x                         6028.8383
# hc_y                         6631.3810
# launch_speed                 5437.4934
# launch_angle                 6260.3193
# if_fielding_alignment        1069.0115
# of_fielding_alignment         377.6502
# xHR                          6359.9901
# xSingle                     12981.8963
# xXBH                         6645.2539

# Generate Probabalistic Predictions 
end_state_pred <- predict(end_state_model, es_test_set, type = "prob")

# Function to extract the four most likely end game states and their probabilities
get_top_4 <- function(prob_vec) {
  top_4_probs <- sort(prob_vec, decreasing = TRUE)[1:4]
  top_4_classes <- names(top_4_probs)
  top_4_probs_and_classes <- matrix(c(top_4_classes, top_4_probs), ncol = 8, byrow = TRUE)
  return(top_4_probs_and_classes)
}

top_4_probs_and_classes <- t(apply(end_state_pred , 1, get_top_4))

colnames(top_4_probs_and_classes) <- c("xES1", "xES2", "xES3","xES4",
                                       "p_ES1", "p_ES2", "p_ES3", "p_ES4")

es_test_set <- cbind(es_test_set,top_4_probs_and_classes ) %>%
  mutate(across(p_ES1:p_ES4, as.numeric))

# Distribution of first guess probability estimates (p_ES1)
#    Min. 1st Qu.  Median 
#  0.1400  0.7160  0.9120 
#  Mean 3rd Qu.    Max. 
#  0.8272  0.9720  1.0000
# The model is very confident in the majority of settings. 
# Lets see the accuracy of the most confident Quartile
test <- es_test_set%>%
  filter(p_ES1 >= 0.9720) %>%
  mutate(correct = ifelse(end_state  == xES1, 
                          1, 0)) %>%
  summarize(correct_guess  = sum(correct) / n(),
            mean = mean(p_ES1))

# Those predictions are 99.15% Correct. 
# The mean expected probability is 98.76%

# What if we check the success rate of the end_state 
# Being one of the three predicted values
test1 <- es_test_set%>%
  mutate(correct = ifelse(end_state  == xES1,
                          1, 0)) %>%
  summarize(number_of_guesses = 1,
            correct_guess  = sum(correct) / n())

test2 <- es_test_set%>%
  mutate(
    correct = ifelse(end_state %in% c(xES1, xES2),
                          1, 0)) %>%
  summarize(number_of_guesses = 2,
            correct_guess  = sum(correct) / n())

test3 <- es_test_set%>%
  mutate(
    correct = ifelse(end_state %in% c(xES1, xES2, xES3),
                          1, 0)) %>%
  summarize(number_of_guesses = 3,
            correct_guess  = sum(correct) / n())

test4 <- es_test_set%>%
  mutate(
    correct = ifelse(end_state %in% c(xES1, xES2, xES3, xES4),
                          1, 0)) %>%
  summarize(number_of_guesses = 4,
            correct_guess  = sum(correct) / n())

test_summaries <- bind_rows(test1, test2, test3, test4)

# Results 
# The model is essentially perfect within 2 guesses.
#|  xES1   |  xES1 & xES2  | xES1 & xES2 & xES3  | xES1 & xES2 & xES3 & xES4|
#| 88.43%  |     99.97%    |      99.983%        |           99.99587 %     |

# Further testing could be done in terms of the accuracy of the probabilities,
# but after sifting through the predictions, it succeeds within
# game states I would expected would be. (double plays, runners tagging, etc.)
# Any error in the first 2 predictions can likely be attributed to fluke 
# plays (losing the ball in the sun, extremely abnormal bounce, extreme weather).
# Errors will also be much higher in outlier environments (Denver)
# Overall, the model is extremely effective. 

# Now, lets simulate some innings.

# Pitcher ID an easily be found on their baseball savant
# webpage, or could be searched using the playerid_lookup()
# function. However, to avoid the bugs of same name players,
# we will use ID

get_pitcher_pbp <- function(id , start_date, end_date){
  
  p <- statcast_search_pitchers(start_date,end_date,pitcherid = id)
  
  p <- p%>%
  mutate(is_bip = ifelse(is.na(hc_x) == TRUE | is.na(launch_angle) == TRUE, FALSE, TRUE))%>%
    filter (events != "", events != "caught_stealing_home", events !="catcher_interf", game_type == "R") %>%
    select(game_pk,at_bat_number, player_name, pitcher, events, is_bip,
           stand, on_1b, on_2b, on_3b, outs_when_up,
           hc_x, hc_y, launch_speed, launch_angle, if_fielding_alignment, of_fielding_alignment)%>%
    rename(start_on_1b = on_1b, start_on_2b = on_2b, start_on_3b = on_3b ) %>%
    mutate(
      start_on_1b = ifelse(is.na(start_on_1b), FALSE, TRUE),
      start_on_2b = ifelse(is.na(start_on_2b), FALSE, TRUE),
      start_on_3b = ifelse(is.na(start_on_3b), FALSE, TRUE))
  
  # define the levels for if_fielding_alignment
  if_levels <- c("", "Infield shift", "Standard", "Strategic")
  # define the levels for of_fielding_alignment
  of_levels <- c("", "4th outfielder", "Standard", "Strategic")

  # create a function that converts a vector to a factor with the desired levels
  to_factor <- function(x, levels) {
    factor(x, levels = levels, exclude = NULL)
  }
  
  # create a data frame with some example data
  data <- tibble(
    if_fielding_alignment = c("", "Infield shift", "Unknown", "Standard"),
    of_fielding_alignment = c("4th outfielder", "Unknown", "Standard", "")
  )
  
  # apply the to_factor function to the if_fielding_alignment and of_fielding_alignment columns
  p$if_fielding_alignment <- to_factor(p$if_fielding_alignment, if_levels)
  p$of_fielding_alignment <- to_factor(p$of_fielding_alignment, of_levels)
  
  
  
  p$stand <- as.factor(p$stand)

  non_bip <- c("walk", "hit_by_pitch", "strikeout", "strikeout_double_play")
  
  p <- p[!(p$events == "field_out" & p$is_bip == FALSE) , ]
  p <- p %>%
    filter(events %in% non_bip | complete.cases(.))
  
 
  return(p)
}


# Now create the functions to simulate Balls in Play

sim_bip <- function(PA){
  
  PA$xHR <- round(as.numeric(predict(hr_model, PA)), 7)
  PA <- cbind(PA, predict( bip_model, PA, type = "prob")) %>%
    rename(xSingle =single, xXBH =  xbh)%>%
    select(-x)
  end_state_pred <- predict(end_state_model, PA, type = "prob")
  top_4_probs<- t(apply(end_state_pred , 1, get_top_4))
  colnames(top_4_probs) <- c("xES1", "xES2", "xES3","xES4", "p_ES1", "p_ES2", "p_ES3", "p_ES4")
  PA <- cbind(PA, top_4_probs) 
  col_idx <- sample.int(4, 1, prob = PA[1, 25:28])
  col_name <- paste0("xES", col_idx)
  PA$sim_end_state <- PA[1, get(col_name)]
  
  PA <- PA %>%
    mutate(end_on_1b = ifelse(substr(sim_end_state, 1, 1) == "1",
                              TRUE, FALSE),
           end_on_2b = ifelse(substr(sim_end_state, 3, 3) == "1",
                              TRUE, FALSE),
           end_on_3b = ifelse(substr(sim_end_state, 5, 5) == "1",
                              TRUE, FALSE),
           outs_created = as.numeric(substr(sim_end_state, 7, 7)),
           runs_scored = as.numeric(substr(sim_end_state, 9, 9)))
  
  PA <- PA %>%
    select(is_bip, events, stand, start_on_1b:outs_when_up, end_on_1b, end_on_2b, 
           end_on_3b, outs_created, runs_scored )
  return(PA)
}

# SIM STRIKOUT/HBP/WALK FUNCTION
sim_non_bip <- function(PA){
  if (PA$events %in% c("walk", "hit_by_pitch")){
    if (!PA$start_on_1b & !PA$start_on_2b & !PA$start_on_3b) {
      PA$end_on_1b <- TRUE
      PA$end_on_2b <- FALSE
      PA$end_on_3b <- FALSE
      PA$outs_created <- 0
      PA$runs_scored <- 0
    }
    if (PA$start_on_1b & !PA$start_on_2b & !PA$start_on_3b) {
      PA$end_on_1b <- TRUE
      PA$end_on_2b <- TRUE
      PA$end_on_3b <- FALSE
      PA$outs_created <- 0
      PA$runs_scored <- 0
    }
    if (!PA$start_on_1b & PA$start_on_2b & !PA$start_on_3b) {
      PA$end_on_1b <- FALSE
      PA$end_on_2b <- TRUE
      PA$end_on_3b <- TRUE
      PA$outs_created <- 0
      PA$runs_scored <- 1
    }
    # 1b false, 2b false, 3b true
    if (!PA$start_on_1b & !PA$start_on_2b & PA$start_on_3b) {
      PA$end_on_1b <- FALSE
      PA$end_on_2b <- FALSE
      PA$end_on_3b <- TRUE
      PA$outs_created <- 0
      PA$runs_scored <- 1
    }
    # 1b true, 2b true, 3b false
    if (PA$start_on_1b & PA$start_on_2b & !PA$start_on_3b) {
      PA$end_on_1b <- TRUE
      PA$end_on_2b <- TRUE
      PA$end_on_3b <- TRUE
      PA$outs_created <- 0
      PA$runs_scored <- 0
    }
    # 1b true, 2b false, 3b true
    if (PA$start_on_1b & !PA$start_on_2b & PA$start_on_3b) {
      PA$end_on_1b <- TRUE
      PA$end_on_2b <- TRUE
      PA$end_on_3b <- TRUE
      PA$outs_created <- 0
      PA$runs_scored <- 0
    }
    # 1b false, 2b true, 3b true
    if (!PA$start_on_1b & PA$start_on_2b & PA$start_on_3b) {
      PA$end_on_1b <- FALSE
      PA$end_on_2b <- TRUE
      PA$end_on_3b <- TRUE
      PA$outs_created <- 0
      PA$runs_scored <- 0
    }
    # Full
    if (PA$start_on_1b & PA$start_on_2b & PA$start_on_3b) {
      PA$end_on_1b <- TRUE
      PA$end_on_2b <- TRUE
      PA$end_on_3b <- TRUE
      PA$outs_created <- 0
      PA$runs_scored <- 1
    }
  }
  if(PA$events %in% c("strikeout", "strikeout_double_play")){
    PA$end_on_1b <- PA$start_on_1b
    PA$end_on_2b <- PA$start_on_2b
    PA$end_on_3b <- PA$start_on_3b
    PA$outs_created <- 1
    PA$runs_scored <- 0
  }
  
  PA <- PA %>%
    select(is_bip, events, stand, start_on_1b:outs_when_up, end_on_1b, end_on_2b, 
           end_on_3b, outs_created, runs_scored )
  
  return(PA)
}


  
# Simulating Seasons
gaus  <- get_pitcher_pbp(592332, "2022-04-07",
                        "2022-10-05")


# Create List of Seasons
# Would preferably create more, but they are computationally 
# expensive to run remote. Currently troubleshooting how
# to run within AWS instead. Unknown how long it would take 
# for normal distribution to emerge.
# using 180IP seasons, as thats how many berrios threw

# Notable issues: setting seed effectively, dealing with extremely rare data points
# with weird missing data (less than 1% of obs), computation time, troubleshooting AWS
# running loop with seperated cores




sim_innings <- function(pitcher_id, start_date, end_date,
                        number_of_innings){
  
  pitcher <- get_pitcher_pbp(pitcher_id, start_date,
                             end_date)
  
  list_of_innings <- list()
  for (num_innings in 1:number_of_innings){
    inning <- tibble(
      is_bip = logical(),
      events = character(),
      stand = factor(levels = c("L", "R")),
      start_on_1b = logical(),
      start_on_2b = logical(),
      start_on_3b = logical(),
      outs_when_up = numeric(),
      end_on_1b = logical(),
      end_on_2b = logical(),
      end_on_3b = logical(),
      outs_created = numeric(),
      runs_scored = numeric()
    )
    
    while (sum(inning$outs_created) < 3 || nrow(inning) == 1) {
      # Acts to create more legitimately random values.
      # Was significant issues regarding repeating simulations.
      set.seed(nrow(inning) +  (seasons * num_innings * as.numeric(substr(format(Sys.time(), "%OS5"), 4, 7))))
      
      # load PA in and initialize
      PA <- pitcher[sample(nrow(pitcher), 1), ]
      
      if(nrow(inning) == 0) {
        PA$start_on_1b <- FALSE
        PA$start_on_2b <- FALSE
        PA$start_on_3b <- FALSE
        PA$outs_when_up <- 0
      } else {
        prev_PA <- inning[nrow(inning), ]
        PA <- pitcher[sample(nrow(pitcher), 1), ]
        PA$start_on_1b <- prev_PA$end_on_1b
        PA$start_on_2b <- prev_PA$end_on_2b
        PA$start_on_3b <- prev_PA$end_on_3b
        PA$outs_when_up <- sum(inning$outs_created)
      }
      
      if (PA$is_bip) {
        PA <- sim_bip(PA)
      } else {
        PA <- sim_non_bip(PA)
      }
      
      inning <- bind_rows(inning, PA)
    }
    
    list_of_innings[[num_innings]] <- inning
  }
  return(list_of_innings)
}


test <- bind_rows(list_of_innings)

inning_summary <- map_dfr(list_of_innings, function(df) {
  runs <- sum(df$runs_scored)
  K <- sum(df$events == "strikeout" | df$events == "strikeout_double_play")
  BB <- sum(df$events == "walk")
  tibble(runs = runs, K = K, BB = BB)
})

bootstrap <- inning_summary %>%
  rep_sample_n(size = 175, reps = 15000, replace = TRUE)%>%
  group_by(replicate) %>%
  summarize(runs  = sum(runs), K = sum(K), BB = sum(BB),
            ERA = (sum(runs)/175) * 9,
            K_per_nine = (K / 175) * 9 )

gausman_innings <- sim_innings(592332, "2022-04-07",
                               "2022-10-05", 300)


gausman_innings2 <- sim_innings(592332, "2022-04-07",
                               "2022-10-05", 150)

gausman <- append(gausman_innings, gausman_innings2 )




gausman_summary <- map_dfr(gausman  , function(df) {
  bip <- df %>%
    mutate(is_hit = ifelse(is_bip == TRUE & outs_created == 0,
                           1, 0)) %>%
    filter(is_bip == TRUE) 
  runs <- sum(df$runs_scored)
  K <- sum(df$events == "strikeout" | df$events == "strikeout_double_play")
  BB <- sum(df$events == "walk")
  HBP <- sum(df$events == "hit_by_pitch")
  BIP <- nrow(bip)
  hits <- sum(bip$is_hit)
  HR <- sum(df$events == "home_run")
  tibble(runs = runs, K = K, BB = BB, HBP = HBP, BIP = BIP,
         hits = hits, HR = HR)
})


gausman_bootstrap <- gausman_summary %>%
  rep_sample_n(size = 175, reps = 15000, replace = TRUE)%>%
  group_by(replicate) %>%
  summarize(runs  = sum(runs), K = sum(K), BB = sum(BB),
            ERA = (sum(runs)/175) * 9,
            FIP  = (((13*sum(HR))+
                       (3*(sum(BB)+sum(HBP)))-(2*sum(K)))/175) + 3.112,
            BABIP = sum(hits) / sum(BIP),
            K_per_nine = (K / 175) * 9 )


ggplot(gausman_bootstrap , aes(x = BABIP)) +
  geom_histogram( fill = "dodgerblue", color = "white") +
  labs(title = "Distribution of BABIP", x = "BABIP", y = "Count") +
  geom_vline(xintercept = 0.363, color = "red", linetype = "dashed") +
  theme_minimal()+
  theme(
    axis.title.x = element_text(size = 10, face = "bold"), 
    axis.title.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 14, face = "bold")
  ) 

ggplot(gausman_bootstrap , aes(x = FIP)) +
  geom_histogram( fill = "dodgerblue", color = "white") +
  labs(title = "Distribution of FIP", x = "FIP", y = "Count") +
  geom_vline(xintercept = 2.40, color = "red", linetype = "dashed") +
  theme_minimal()+
  theme(
    axis.title.x = element_text(size = 10, face = "bold"), 
    axis.title.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 14, face = "bold")
  ) 


ggplot(verlander_bootstrap) +
  geom_density(aes(x = FIP), alpha = 0.6, color = "#006BA6", fill = "#006BA6") +
  geom_density(aes(x = ERA), alpha = 0.6, color = "#DA291C", fill = "#DA291C") +
  labs(title = "Distribution of FIP/ERA", x = "FIP/ERA", y = "Count") +
  theme_minimal()+
  theme(
    axis.title.x = element_text(size = 10, face = "bold"), 
    axis.title.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 14, face = "bold")
  ) 



verlander <- sim_innings(pitcher_id = 434378,start_date = "2022-04-07",
                          end_date = "2022-10-05", number_of_innings = 4375) 


verlander_summary <- map_dfr(verlander  , function(df) {
  bip <- df %>%
    mutate(is_hit = ifelse(is_bip == TRUE & outs_created == 0,
                           1, 0)) %>%
    filter(is_bip == TRUE) 
  runs <- sum(df$runs_scored)
  K <- sum(df$events == "strikeout" | df$events == "strikeout_double_play")
  BB <- sum(df$events == "walk")
  HBP <- sum(df$events == "hit_by_pitch")
  BIP <- nrow(bip)
  hits <- sum(bip$is_hit)
  HR <- sum(df$events == "home_run")
  tibble(runs = runs, K = K, BB = BB, HBP = HBP, BIP = BIP,
         hits = hits, HR = HR)
})


ggplot(data = verlander_bootstrap) +
  geom_point(aes(x = ERA, y= BABIP), alpha = 0.05, color = "red")+
  geom_smooth(aes(x = ERA, y= BABIP), method = "lm", color ="#D3D3D3")+
  labs(title = "Justin Verlander Bootstrapped ERA vs. BABIP")+
  theme_minimal()+
  theme(
    axis.title.x = element_text(size = 10, face = "bold"), 
    axis.title.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 14, face = "bold")
  ) 

basic_model <- lm(ERA ~ BABIP, data = verlander_bootstrap)
summary(basic_model)












