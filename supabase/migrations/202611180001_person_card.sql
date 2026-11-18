ALTER TABLE cams_persons ADD COLUMN IF NOT EXISTS card_uid TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS cams_persons_card ON cams_persons(card_uid);
