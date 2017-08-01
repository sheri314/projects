import time
from selenium import webdriver
from login import login
import sys

search = sys.argv[1]
driver = webdriver.Chrome('/usr/local/bin/chromedriver')  # Optional argument, if not specified will search path.
driver.get('https://twitter.com/login')

# here is where some useful work would typically happen

userfield = driver.find_element_by_css_selector('.js-username-field.email-input.js-initial-focus')
passwordfield = driver.find_element_by_css_selector('.js-password-field')

username, password = login()
userfield.send_keys(username)
passwordfield.send_keys(password)
passwordfield.submit()

driver.get("https://twitter.com/search-advanced")

searchterm = driver.find_element_by_xpath("//*[@id='page-container']/div/div[1]/form/fieldset[1]/div[1]/label/input")

searchterm.send_keys(search)

searchhtags = driver.find_element_by_css_selector()

submit = driver.find_element_by_css_selector("button.button.btn.primary-btn.submit.selected")
submit.click()

raw_input("Press Enter to quit")
driver.quit() # close browser
