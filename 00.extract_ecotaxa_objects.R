#--------------------------------------------------------------------------#
# Project: Detectron2_plankton_training
# Script purpose: Extract manual masks objects from EcoTaxa, make a taxo tree count to keep only objects in relevant classes, and save table.
# Date: 20/01/2021
# Author: Thelma Panaiotis
#--------------------------------------------------------------------------#


library(tidyverse)
library(ecotaxar)
library(data.tree)
source("lib/my_gs4_auth.R") # for google sheet authentication


## Connect to database
db <- db_connect_ecotaxa()

## Extract objects from cc4 manual masks (project 2769) ----
#--------------------------------------------------------------------------#
# NB: cc4 extraction is done apart from others projects because bbox values are labelled differently in project 2769
# "bbox1", "bbox2", "bbox3", "bbox4" in project 2769
# "bbox-0", "bbox-1", "bbox-2", "bbox-3" in other projects (prelag1 3394, lag1 3393 and cc2 3392)

# Project cc4
my_projects <- as.integer(2769)

# Project to extract
proj <- tbl(db, "projects") %>% filter(projid %in% my_projects) %>% collect()

# Extract taxo
taxo <- tbl(db, "taxonomy") %>% collect() 

# Extract objects
obj_cc4 <- tbl(db, "objects") %>% 
  filter(projid %in% my_projects, classif_qual == "V") %>% 
  map_names(mapping=proj$mappingobj) %>% 
  select(
    object_id = orig_id, classif_id, 
    area, 
    convex_area, 
    filled_area, 
    eccentricity, 
    mean_intensity, 
    perimeter, 
    orientation,
    bbox0=bbox1,
    bbox1=bbox2,
    bbox2=bbox3,
    bbox3=bbox4,
  ) %>% 
  collect() %>% 
  mutate(
    taxon=taxo_name(classif_id, taxo=taxo, unique=TRUE),
    lineage=lineage(classif_id, taxo=taxo)
  ) %>% 
  select(-classif_id)


## Extract objects from other projects ----
#--------------------------------------------------------------------------#
# cc2 is 3392
# lag1 is 3393
# prelag1 is 3394

# Other projects
my_projects <- c(as.integer(3392), as.integer(3393), as.integer(3394))

# Project to extract
proj <- tbl(db, "projects") %>% filter(projid %in% my_projects) %>% collect()

# Extract taxo
taxo <- tbl(db, "taxonomy") %>% collect() 

# Extract objects
obj_oth <- tbl(db, "objects") %>% 
  filter(projid %in% my_projects, classif_qual == "V") %>% 
  map_names(mapping=proj$mappingobj) %>% 
  select(
    object_id = orig_id, classif_id, 
    area, 
    convex_area, 
    filled_area, 
    eccentricity, 
    mean_intensity, 
    perimeter, 
    orientation,
    bbox0=`bbox-0`,
    bbox1=`bbox-1`,
    bbox2=`bbox-2`,
    bbox3=`bbox-3`,
  ) %>% 
  collect() %>% 
  mutate(
    taxon=taxo_name(classif_id, taxo=taxo, unique=TRUE),
    lineage=lineage(classif_id, taxo=taxo)
  ) %>% 
  select(-classif_id)

# Disconnect from database
db_disconnect_ecotaxa(db)

## Merge tables from cc4 and other projects ----
#--------------------------------------------------------------------------#
# Merge tables
obj <- union(obj_oth, obj_cc4)
# List taxa
taxa <- obj %>% select(lineage, taxon) %>% unique()


## Make a taxo tree count to select only objects in relevant taxa ----
#--------------------------------------------------------------------------#
ss <- 'https://docs.google.com/spreadsheets/d/154LfaN1eguPQnCWnzmw94ukV2I9vvfo65uJH37ni1SI/edit#gid=1657818463'

# Read existent sheet
current_tcd <- read_sheet(ss)

# Build the tree
tc <- count(obj, taxon, lineage) %>% 
  # convert it into a tree
  rename(pathString=lineage) %>%
  arrange(pathString) %>%
  as.Node()

print(tc, "taxon","n", limit = 50)
# Convert to dataframe
tcd <- ToDataFrameTree(tc, "taxon", "n")%>% 
  as_tibble() %>% 
  rename(level0=taxon, nb_level0=n)

## Write tree into GSS
# Start by erasing previous data (3 first columns) in spreadsheet
range_flood(ss, sheet = "tcd", range = "tcd!A:C", reformat = FALSE)
# Write new tree
range_write(ss, data = tcd) 
# Open it in browser tab, make changes if needed
gs4_browse(ss)


## Rematch taxo with GSS ----
#--------------------------------------------------------------------------#
# Read match between old and new taxo and generate a classif_id
taxo_match <- read_sheet(ss) %>% 
  select(level0, level1) %>% 
  filter(!is.na(level0)) %>% 
  rename(taxon = level0, new_taxon = level1) %>% 
  arrange(new_taxon) %>% 
  mutate(classif_id = row_number()-1)

# Correct taxo on objects table
obj <- obj %>% 
  left_join(taxo_match, by = "taxon") %>% 
  filter(!is.na(new_taxon)) %>% 
  select(-c(taxon, lineage)) %>% 
  rename(taxon = new_taxon) 


## Write table ----
#--------------------------------------------------------------------------#
# Write to csv
write_csv(obj, file = "data/ecotaxa_export.csv")


